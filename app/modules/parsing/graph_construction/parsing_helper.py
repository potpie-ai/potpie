import math
import time
import warnings
from collections import Counter, defaultdict, namedtuple
from pathlib import Path

from grep_ast import TreeContext, filename_to_lang
from pygments.lexers import guess_lexer_for_filename
from pygments.token import Token
from pygments.util import ClassNotFound
from tqdm import tqdm
from tree_sitter_languages import get_language, get_parser  # noqa: E402
import json
import logging
import os
import shutil
import tarfile
import requests
from fastapi import HTTPException
from git import Repo, GitCommandError
from uuid6 import uuid7
from app.modules.projects.projects_schema import ProjectStatusEnum
from app.modules.projects.projects_service import ProjectService
import networkx as nx
from sqlalchemy.orm import Session 

# tree_sitter is throwing a FutureWarning
warnings.simplefilter("ignore", category=FutureWarning)
Tag = namedtuple("Tag", "rel_fname fname line end_line name kind type".split())

class ParsingServiceError(Exception):
    """Base exception class for ParsingService errors."""

class ParsingFailedError(ParsingServiceError):
    """Raised when a parsing fails."""
    
class ParseHelper:
    def __init__(self, db_session: Session):
        self.project_manager = ProjectService(db_session) 
        self.db = db_session

    def download_and_extract_tarball(self, repo, branch, target_dir, auth, repo_details, user_id):
        try:
            tarball_url = repo_details.get_archive_link("tarball", branch)
            response = requests.get(
                tarball_url,
                stream=True,
                headers={"Authorization": f"{auth.token}"},
            )
            response.raise_for_status()  # Check for request errors
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching tarball: {e}")
            return e

        tarball_path = os.path.join(target_dir, f"{repo.full_name.replace('/', '-')}-{branch}.tar.gz")
        try:
            with open(tarball_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        except IOError as e:
            logging.error(f"Error writing tarball to file: {e}")
            return e

        final_dir = os.path.join(target_dir, f"{repo.full_name.replace('/', '-')}-{branch}-{user_id}")
        try:
            with tarfile.open(tarball_path, "r:gz") as tar:
                for member in tar.getmembers():
                    member_path = os.path.join(
                        final_dir,
                        os.path.relpath(member.name, start=member.name.split("/")[0]),
                    )
                    if member.isdir():
                        os.makedirs(member_path, exist_ok=True)
                    else:
                        member_dir = os.path.dirname(member_path)
                        if not os.path.exists(member_dir):
                            os.makedirs(member_dir)
                        with open(member_path, "wb") as f:
                            if member.size > 0:
                                f.write(tar.extractfile(member).read())
        except (tarfile.TarError, IOError) as e:
            logging.error(f"Error extracting tarball: {e}")
            return e

        try:
            os.remove(tarball_path)
        except OSError as e:
            logging.error(f"Error removing tarball: {e}")
            return e

        return final_dir

    @staticmethod
    def detect_repo_language(repo_dir):
        lang_count = {
            "c_sharp": 0, "c": 0, "cpp": 0, "elisp": 0, "elixir": 0, "elm": 0,
            "go": 0, "java": 0, "javascript": 0, "ocaml": 0, "php": 0, "python": 0,
            "ql": 0, "ruby": 0, "rust": 0, "typescript": 0, "other": 0
        }
        total_chars = 0
        for root, _, files in os.walk(repo_dir):
            for file in files:
                file_path = os.path.join(root, file)
                ext = os.path.splitext(file)[1].lower()
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        total_chars += len(content)
                        if ext == '.cs':
                            lang_count["c_sharp"] += 1
                        elif ext == '.c':
                            lang_count["c"] += 1
                        elif ext in ['.cpp', '.cxx', '.cc']:
                            lang_count["cpp"] += 1
                        elif ext == '.el':
                            lang_count["elisp"] += 1
                        elif ext == '.ex' or ext == '.exs':
                            lang_count["elixir"] += 1
                        elif ext == '.elm':
                            lang_count["elm"] += 1
                        elif ext == '.go':
                            lang_count["go"] += 1
                        elif ext == '.java':
                            lang_count["java"] += 1
                        elif ext in ['.js', '.jsx']:
                            lang_count["javascript"] += 1
                        elif ext == '.ml' or ext == '.mli':
                            lang_count["ocaml"] += 1
                        elif ext == '.php':
                            lang_count["php"] += 1
                        elif ext == '.py':
                            lang_count["python"] += 1
                        elif ext == '.ql':
                            lang_count["ql"] += 1
                        elif ext == '.rb':
                            lang_count["ruby"] += 1
                        elif ext == '.rs':
                            lang_count["rust"] += 1
                        elif ext in ['.ts', '.tsx']:
                            lang_count["typescript"] += 1
                        else:
                            lang_count["other"] += 1
                except (UnicodeDecodeError, FileNotFoundError):
                    continue
        # Determine the predominant language based on counts
        predominant_language = max(lang_count, key=lang_count.get)
        return predominant_language if lang_count[predominant_language] > 0 else "other"
    
    
    async def setup_project_directory(
        self,  repo, branch, auth, repo_details, user_id, project_id = None # Change type to str
    ):

        if not project_id:
            pid = str(uuid7())
            project_id = await self.project_manager.register_project(
                f"{repo.full_name}",
            branch,
            user_id,
            pid,
        )
        
        
        if isinstance(repo_details, Repo):
            extracted_dir = repo_details.working_tree_dir
            try:
                current_dir = os.getcwd()
                os.chdir(extracted_dir)  # Change to the cloned repo directory
                repo_details.git.checkout(branch)
            except GitCommandError as e:
                logging.error(f"Error checking out branch: {e}")
                raise HTTPException(
                    status_code=400, detail=f"Failed to checkout branch {branch}"
                )
            finally:
                os.chdir(current_dir)  # Restore the original working directory
            branch_details = repo_details.head.commit
            latest_commit_sha = branch_details.hexsha
        else:
            
            extracted_dir = self.download_and_extract_tarball(
                repo, branch, os.getenv("PROJECT_PATH"), auth, repo_details, user_id
            )
            branch_details = repo_details.get_branch(branch)
            latest_commit_sha = branch_details.commit.sha

        repo_metadata = ParseHelper.extract_repository_metadata(repo_details)
        repo_metadata["error_message"] = None
        project_metadata = json.dumps(repo_metadata).encode("utf-8")
        ProjectService.update_project(self.db, project_id, properties=project_metadata, commit_id=latest_commit_sha, status=ProjectStatusEnum.CLONED.value)

        return extracted_dir, project_id
    
    def extract_repository_metadata(repo):
        if isinstance(repo, Repo):
            metadata = ParseHelper.extract_local_repo_metadata(repo)
        else:
            metadata = ParseHelper.extract_remote_repo_metadata(repo)
        return metadata

    def extract_local_repo_metadata(repo):
        languages = ParseHelper.get_local_repo_languages(repo.working_tree_dir)
        total_bytes = sum(languages.values())

        metadata = {
            "basic_info": {
                "full_name": os.path.basename(repo.working_tree_dir),
                "description": None,
                "created_at": None,
                "updated_at": None,
                "default_branch": repo.head.ref.name,
            },
            "metrics": {
                "size": ParseHelper.get_directory_size(repo.working_tree_dir),
                "stars": None,
                "forks": None,
                "watchers": None,
                "open_issues": None,
            },
            "languages": {
                "breakdown": languages,
                "total_bytes": total_bytes,
            },
            "commit_info": {
                "total_commits": len(list(repo.iter_commits()))
            },
            "contributors": {
                "count": len(list(repo.iter_commits('--all'))),
            },
            "topics": [],
        }

        return metadata

    def get_local_repo_languages(path):
        total_bytes = 0
        python_bytes = 0

        for dirpath, _, filenames in os.walk(path):
            for filename in filenames:
                file_extension = os.path.splitext(filename)[1]
                file_path = os.path.join(dirpath, filename)
                file_size = os.path.getsize(file_path)
                total_bytes += file_size
                if file_extension == '.py':
                    python_bytes += file_size

        languages = {}
        if total_bytes > 0:
            languages['Python'] = python_bytes
            languages['Other'] = total_bytes - python_bytes

        return languages

    def extract_remote_repo_metadata(repo):
        languages = repo.get_languages()
        total_bytes = sum(languages.values())

        metadata = {
            "basic_info": {
                "full_name": repo.full_name,
                "description": repo.description,
                "created_at": repo.created_at.isoformat(),
                "updated_at": repo.updated_at.isoformat(),
                "default_branch": repo.default_branch,
            },
            "metrics": {
                "size": repo.size,
                "stars": repo.stargazers_count,
                "forks": repo.forks_count,
                "watchers": repo.watchers_count,
                "open_issues": repo.open_issues_count,
            },
            "languages": {
                "breakdown": languages,
                "total_bytes": total_bytes,
            },
            "commit_info": {
                "total_commits": repo.get_commits().totalCount
            },
            "contributors": {
                "count": repo.get_contributors().totalCount,
            },
            "topics": repo.get_topics(),
        }

        return metadata


class RepoMap:


    warned_files = set()

    def __init__(
        self,
        map_tokens=1024,
        root=None,
        main_model=None,
        io=None,
        repo_content_prefix=None,
        verbose=False,
        max_context_window=None,
        map_mul_no_files=8,
    ):
        self.io = io
        self.verbose = verbose

        if not root:
            root = os.getcwd()
        self.root = root

     

        self.max_map_tokens = map_tokens
        self.map_mul_no_files = map_mul_no_files
        self.max_context_window = max_context_window

            
        self.repo_content_prefix = repo_content_prefix

    def get_repo_map(self, chat_files, other_files, mentioned_fnames=None, mentioned_idents=None):
        if self.max_map_tokens <= 0:
            return
        if not other_files:
            return
        if not mentioned_fnames:
            mentioned_fnames = set()
        if not mentioned_idents:
            mentioned_idents = set()

        max_map_tokens = self.max_map_tokens

        # With no files in the chat, give a bigger view of the entire repo
        padding = 4096
        if max_map_tokens and self.max_context_window:
            target = min(
                max_map_tokens * self.map_mul_no_files,
                self.max_context_window - padding,
            )
        else:
            target = 0
        if not chat_files and self.max_context_window and target > 0:
            max_map_tokens = target

        try:
            files_listing = self.get_ranked_tags_map(
                chat_files, other_files, max_map_tokens, mentioned_fnames, mentioned_idents
            )
        except RecursionError:
            self.io.tool_error("Disabling repo map, git repo too large?")
            self.max_map_tokens = 0
            return

        if not files_listing:
            return

        num_tokens = self.token_count(files_listing)
        if self.verbose:
            self.io.tool_output(f"Repo-map: {num_tokens / 1024:.1f} k-tokens")

        if chat_files:
            other = "other "
        else:
            other = ""

        if self.repo_content_prefix:
            repo_content = self.repo_content_prefix.format(other=other)
        else:
            repo_content = ""

        repo_content += files_listing

        return repo_content

    def get_rel_fname(self, fname):
        return os.path.relpath(fname, self.root)

    def split_path(self, path):
        path = os.path.relpath(path, self.root)
        return [path + ":"]


    def save_tags_cache(self):
        pass

    def get_mtime(self, fname):
        try:
            return os.path.getmtime(fname)
        except FileNotFoundError:
            self.io.tool_error(f"File not found error: {fname}")

    def get_tags(self, fname, rel_fname):
        # Check if the file is in the cache and if the modification time has not changed
        file_mtime = self.get_mtime(fname)
        if file_mtime is None:
            return []

        data = list(self.get_tags_raw(fname, rel_fname))


        return data

    def get_tags_raw(self, fname, rel_fname):
        lang = filename_to_lang(fname)
        if not lang:
            return

        language = get_language(lang)
        parser = get_parser(lang)

        query_scm = get_scm_fname(lang)
        if not query_scm.exists():
            return
        query_scm = query_scm.read_text()

        code = self.io.read_text(fname)
        if not code:
            return
        tree = parser.parse(bytes(code, "utf-8"))

        # Run the tags queries
        query = language.query(query_scm)
        captures = query.captures(tree.root_node)

        captures = list(captures)

        saw = set()
        for node, tag in captures:
            if tag.startswith("name.definition."):
                kind = "def"
                type = tag.split(".")[-1]  # 
            elif tag.startswith("name.reference."):
                kind = "ref"
                type = tag.split(".")[-1]  # 
            else:
                continue

            saw.add(kind)

            result = Tag(
                rel_fname=rel_fname,
                fname=fname,
                name=node.text.decode("utf-8"),
                kind=kind,
                line=node.start_point[0],
                end_line=node.end_point[0],
                type=type,
            )

            yield result

        if "ref" in saw:
            return
        if "def" not in saw:
            return

        # We saw defs, without any refs
        # Some tags files only provide defs (cpp, for example)
        # Use pygments to backfill refs

        try:
            lexer = guess_lexer_for_filename(fname, code)
        except ClassNotFound:
            return

        tokens = list(lexer.get_tokens(code))
        tokens = [token[1] for token in tokens if token[0] in Token.Name]

        for token in tokens:
            yield Tag(
                rel_fname=rel_fname,
                fname=fname,
                name=token,
                kind="ref",
                line=-1,
                end_line=-1,
                type="unknown",
            )

    def get_ranked_tags(self, chat_fnames, other_fnames, mentioned_fnames, mentioned_idents):
        defines = defaultdict(set)
        references = defaultdict(list)
        definitions = defaultdict(set)

        personalization = dict()

        fnames = set(chat_fnames).union(set(other_fnames))
        chat_rel_fnames = set()

        fnames = sorted(fnames)

        # Default personalization for unspecified files is 1/num_nodes
        # https://networkx.org/documentation/stable/_modules/networkx/algorithms/link_analysis/pagerank_alg.html#pagerank
        personalize = 100 / len(fnames)

        
        fnames = tqdm(fnames)
        

        for fname in fnames:
            if not Path(fname).is_file():
                if fname not in self.warned_files:
                    if Path(fname).exists():
                        self.io.tool_error(
                            f"Repo-map can't include {fname}, it is not a normal file"
                        )
                    else:
                        self.io.tool_error(f"Repo-map can't include {fname}, it no longer exists")

                self.warned_files.add(fname)
                continue

            # dump(fname)
            rel_fname = self.get_rel_fname(fname)

            if fname in chat_fnames:
                personalization[rel_fname] = personalize
                chat_rel_fnames.add(rel_fname)

            if rel_fname in mentioned_fnames:
                personalization[rel_fname] = personalize

            tags = list(self.get_tags(fname, rel_fname))
            if tags is None:
                continue

            for tag in tags:
                if tag.kind == "def":
                    defines[tag.name].add(rel_fname)
                    key = (rel_fname, tag.name)
                    definitions[key].add(tag)

                if tag.kind == "ref":
                    references[tag.name].append(rel_fname)

        ##
        # dump(defines)
        # dump(references)
        # dump(personalization)

        if not references:
            references = dict((k, list(v)) for k, v in defines.items())

        idents = set(defines.keys()).intersection(set(references.keys()))

        G = nx.MultiDiGraph()

        for ident in idents:
            definers = defines[ident]
            if ident in mentioned_idents:
                mul = 10
            elif ident.startswith("_"):
                mul = 0.1
            else:
                mul = 1

            for referencer, num_refs in Counter(references[ident]).items():
                for definer in definers:
                    # dump(referencer, definer, num_refs, mul)
                    # if referencer == definer:
                    #    continue

                    # scale down so high freq (low value) mentions don't dominate
                    num_refs = math.sqrt(num_refs)

                    G.add_edge(referencer, definer, weight=mul * num_refs, ident=ident)

        if not references:
            pass

        if personalization:
            pers_args = dict(personalization=personalization, dangling=personalization)
        else:
            pers_args = dict()

        try:
            ranked = nx.pagerank(G, weight="weight", **pers_args)
        except ZeroDivisionError:
            return []

        # distribute the rank from each source node, across all of its out edges
        ranked_definitions = defaultdict(float)
        for src in G.nodes:
            src_rank = ranked[src]
            total_weight = sum(data["weight"] for _src, _dst, data in G.out_edges(src, data=True))
            # dump(src, src_rank, total_weight)
            for _src, dst, data in G.out_edges(src, data=True):
                data["rank"] = src_rank * data["weight"] / total_weight
                ident = data["ident"]
                ranked_definitions[(dst, ident)] += data["rank"]

        ranked_tags = []
        ranked_definitions = sorted(ranked_definitions.items(), reverse=True, key=lambda x: x[1])

        # dump(ranked_definitions)

        for (fname, ident), rank in ranked_definitions:
            
            if fname in chat_rel_fnames:
                continue
            ranked_tags += list(definitions.get((fname, ident), []))

        rel_other_fnames_without_tags = set(self.get_rel_fname(fname) for fname in other_fnames)

        fnames_already_included = set(rt[0] for rt in ranked_tags)

        top_rank = sorted([(rank, node) for (node, rank) in ranked.items()], reverse=True)
        for rank, fname in top_rank:
            if fname in rel_other_fnames_without_tags:
                rel_other_fnames_without_tags.remove(fname)
            if fname not in fnames_already_included:
                ranked_tags.append((fname,))

        for fname in rel_other_fnames_without_tags:
            ranked_tags.append((fname,))

        return ranked_tags

    def get_ranked_tags_map(
        self,
        chat_fnames,
        other_fnames=None,
        max_map_tokens=None,
        mentioned_fnames=None,
        mentioned_idents=None,
    ):
        if not other_fnames:
            other_fnames = list()
        if not max_map_tokens:
            max_map_tokens = self.max_map_tokens
        if not mentioned_fnames:
            mentioned_fnames = set()
        if not mentioned_idents:
            mentioned_idents = set()

        ranked_tags = self.get_ranked_tags(
            chat_fnames, other_fnames, mentioned_fnames, mentioned_idents
        )

        num_tags = len(ranked_tags)
        lower_bound = 0
        upper_bound = num_tags
        best_tree = None
        best_tree_tokens = 0

        chat_rel_fnames = [self.get_rel_fname(fname) for fname in chat_fnames]

        # Guess a small starting number to help with giant repos
        middle = min(max_map_tokens // 25, num_tags)

        self.tree_cache = dict()

        while lower_bound <= upper_bound:
            tree = self.to_tree(ranked_tags[:middle], chat_rel_fnames)
            num_tokens = self.token_count(tree)

            if num_tokens < max_map_tokens and num_tokens > best_tree_tokens:
                best_tree = tree
                best_tree_tokens = num_tokens

            if num_tokens < max_map_tokens:
                lower_bound = middle + 1
            else:
                upper_bound = middle - 1

            middle = (lower_bound + upper_bound) // 2

        return best_tree

    tree_cache = dict()

    def render_tree(self, abs_fname, rel_fname, lois):
        key = (rel_fname, tuple(sorted(lois)))

        if key in self.tree_cache:
            return self.tree_cache[key]

        code = self.io.read_text(abs_fname) or ""
        if not code.endswith("\n"):
            code += "\n"

        context = TreeContext(
                rel_fname,
                code,
                color=False,
                line_number=False,
                child_context=False,
                last_line=False,
                margin=0,
                mark_lois=False,
                loi_pad=0,
                show_top_of_file_parent_scope=False,
            )

        for start, end in lois:
            context.add_lines_of_interest(range(start, end + 1))
        context.add_context()
        res = context.format()
        self.tree_cache[key] = res
        return res


    def create_graph(self, repo_dir):
        
        start_time = time.time()
        logging.info("Starting parsing of codebase")  # Log start

        G = nx.MultiDiGraph()
        defines = defaultdict(list)
        references = defaultdict(list)
        file_count = 0  # Initialize file counter
        for root, _, files in os.walk(repo_dir):
            for file in files:
                file_count += 1  # Increment file counter
                logging.info(f"Processing file number: {file_count}")  # Log file number
                
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, repo_dir)
                
                if not self.is_text_file(file_path):
                    continue
                
                tags = self.get_tags(file_path, rel_path)
                
                current_class = None
                current_function = None
                for tag in tags:
                    if tag.kind == "def":
                        if tag.type == "class":
                            current_class = tag.name
                            current_function = None
                        elif tag.type == "function":
                            current_function = tag.name
                        node_name = f"{current_class}.{tag.name}@{rel_path}" if current_class else f"{rel_path}:{tag.name}"
                        defines[tag.name].append((node_name, tag.line, tag.end_line, tag.type, rel_path, current_class))
                        G.add_node(node_name, file=rel_path, line=tag.line, end_line=tag.end_line, type=tag.type)
                    elif tag.kind == "ref":
                        source = f"{current_class}.{current_function}@{rel_path}" if current_class and current_function else f"{rel_path}:{current_function}" if current_function else rel_path
                        references[tag.name].append((source, tag.line, tag.end_line, tag.type, rel_path, current_class))
        
        # Create edges
        for ident, refs in references.items():
            if ident in defines:
                if len(defines[ident]) == 1:  # Unique definition
                    target, def_line, end_def_line, def_type, def_file, def_class = defines[ident][0]
                    for (source, ref_line, end_ref_line, ref_type, ref_file, ref_class) in refs:
                        G.add_edge(source, target, type=ref_type, ident=ident, ref_line=ref_line, end_ref_line=end_ref_line, def_line=def_line, end_def_line=end_def_line)
                else:  # Apply scoring system for non-unique definitions
                    for (source, ref_line, end_ref_line, ref_type, ref_file, ref_class) in refs:
                        best_match = None
                        best_match_score = -1
                        for (target, def_line, end_def_line, def_type, def_file, def_class) in defines[ident]:
                            if source != target:  # Avoid self-references
                                match_score = 0
                                if ref_file == def_file:
                                    match_score += 2
                                elif os.path.dirname(ref_file) == os.path.dirname(def_file):
                                    match_score += 1  # Add a point for being in the same directory
                                if ref_class == def_class:
                                    match_score += 1
                                if match_score > best_match_score:
                                    best_match = (target, def_line, end_def_line, def_type)
                                    best_match_score = match_score
                
                        if best_match:
                            target, def_line, end_def_line, def_type = best_match
                            G.add_edge(source, target, type=ref_type, ident=ident, ref_line=ref_line, end_ref_line=end_ref_line, def_line=def_line, end_def_line=end_def_line)

        end_time = time.time()
        logging.info(f"Parsing completed, time taken: {end_time - start_time} seconds")  # Log end
        return G
    
    def is_text_file(self, file_path):
        # Simple check to determine if a file is likely to be a text file
        # You might want to expand this based on your specific needs
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                f.read(1024)
            return True
        except UnicodeDecodeError:
            return False
    def to_tree(self, tags, chat_rel_fnames):
        if not tags:
            return ""

        tags = [tag for tag in tags if tag[0] not in chat_rel_fnames]
        tags = sorted(tags)

        cur_fname = None
        cur_abs_fname = None
        lois = None
        output = ""

        # add a bogus tag at the end so we trip the this_fname != cur_fname...
        dummy_tag = (None,)
        for tag in tags + [dummy_tag]:
            this_rel_fname = tag[0]

            # ... here ... to output the final real entry in the list
            if this_rel_fname != cur_fname:
                if lois is not None:
                    output += "\n"
                    output += cur_fname + ":\n"
                    output += self.render_tree(cur_abs_fname, cur_fname, lois)
                    lois = None
                elif cur_fname:
                    output += "\n" + cur_fname + "\n"
                if type(tag) is Tag:
                    lois = []
                    cur_abs_fname = tag.fname
                cur_fname = this_rel_fname

            if lois is not None:
                lois.append((tag.line, tag.end_line))

        # truncate long lines, in case we get minified js or something else crazy
        output = "\n".join([line[:100] for line in output.splitlines()]) + "\n"

        return output





def get_scm_fname(lang):
    # Load the tags queries
    try:
        return Path(os.path.dirname(__file__)).joinpath("queries", f"tree-sitter-{lang}-tags.scm")
    except KeyError:
        return


