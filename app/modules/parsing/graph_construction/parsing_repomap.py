import logging
import math
import os
import time
import warnings
from collections import Counter, defaultdict, namedtuple
from pathlib import Path

import networkx as nx
from grep_ast import TreeContext, filename_to_lang
from pygments.lexers import guess_lexer_for_filename
from pygments.token import Token
from pygments.util import ClassNotFound
from tqdm import tqdm
from tree_sitter import Parser
from tree_sitter_languages import get_language, get_parser  # noqa: E402

# tree_sitter is throwing a FutureWarning
warnings.simplefilter("ignore", category=FutureWarning)
Tag = namedtuple("Tag", "rel_fname fname line end_line name kind type".split())


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

    def get_repo_map(
        self, chat_files, other_files, mentioned_fnames=None, mentioned_idents=None
    ):
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
                chat_files,
                other_files,
                max_map_tokens,
                mentioned_fnames,
                mentioned_idents,
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

    @staticmethod
    def get_tags_from_code(fname, code):
        lang = filename_to_lang(fname)
        if not lang:
            return

        language = get_language(lang)
        parser = get_parser(lang)

        query_scm = get_scm_fname(lang)
        if not query_scm.exists():
            return
        query_scm = query_scm.read_text()

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
                rel_fname=fname,
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
                rel_fname=fname,
                fname=fname,
                name=token,
                kind="ref",
                line=-1,
                end_line=-1,
                type="unknown",
            )

    def get_ranked_tags(
        self, chat_fnames, other_fnames, mentioned_fnames, mentioned_idents
    ):
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
                        self.io.tool_error(
                            f"Repo-map can't include {fname}, it no longer exists"
                        )

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
            total_weight = sum(
                data["weight"] for _src, _dst, data in G.out_edges(src, data=True)
            )
            # dump(src, src_rank, total_weight)
            for _src, dst, data in G.out_edges(src, data=True):
                data["rank"] = src_rank * data["weight"] / total_weight
                ident = data["ident"]
                ranked_definitions[(dst, ident)] += data["rank"]

        ranked_tags = []
        ranked_definitions = sorted(
            ranked_definitions.items(), reverse=True, key=lambda x: x[1]
        )

        # dump(ranked_definitions)

        for (fname, ident), rank in ranked_definitions:
            if fname in chat_rel_fnames:
                continue
            ranked_tags += list(definitions.get((fname, ident), []))

        rel_other_fnames_without_tags = set(
            self.get_rel_fname(fname) for fname in other_fnames
        )

        fnames_already_included = set(rt[0] for rt in ranked_tags)

        top_rank = sorted(
            [(rank, node) for (node, rank) in ranked.items()], reverse=True
        )
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
        logging.info("Starting parsing of codebase")

        G = nx.MultiDiGraph()
        defines = defaultdict(list)
        references = defaultdict(list)
        file_count = 0

        for root, dirs, files in os.walk(repo_dir):
            # Ignore folders starting with '.'
            if any(part.startswith(".") for part in root.split(os.sep)):
                continue

            for file in files:
                file_count += 1
                logging.info(f"Processing file number: {file_count}")

                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, repo_dir)

                if not self.is_text_file(file_path):
                    continue

                tags = self.get_tags(file_path, rel_path)

                # Extract full file content
                file_content = self.io.read_text(file_path) or ""
                if not file_content.endswith("\n"):
                    file_content += "\n"

                # Parse the file using tree-sitter
                language = RepoMap.get_language_for_file(file_path)
                if language:
                    parser = Parser()
                    parser.set_language(language)
                    tree = parser.parse(bytes(file_content, "utf8"))
                    root_node = tree.root_node

                current_class = None
                current_function = None
                for tag in tags:
                    if tag.kind == "def":
                        if tag.type == "class":
                            current_class = tag.name
                            current_function = None
                            node_type = "class"
                        elif tag.type == "function":
                            current_function = tag.name
                            node_type = "function"
                        else:
                            node_type = "other"

                        node_name = f"{rel_path}:{tag.name}"

                        # Extract code for the current tag using AST
                        if language:
                            node = RepoMap.find_node_by_range(
                                root_node, tag.line, node_type
                            )
                            if node:
                                code_context = file_content[
                                    node.start_byte : node.end_byte
                                ]
                                node_end_line = (
                                    node.end_point[0] + 1
                                )  # Adding 1 to match 1-based line numbering
                            else:
                                code_context = ""
                                node_end_line = tag.end_line
                                continue
                        else:
                            code_context = ""
                            node_end_line = tag.end_line
                            continue

                        defines[tag.name].append(
                            (
                                node_name,
                                tag.line,
                                node_end_line,
                                node_type,
                                rel_path,
                                current_class,
                            )
                        )
                        G.add_node(
                            node_name,
                            file=rel_path,
                            line=tag.line,
                            end_line=node_end_line,
                            type=tag.type,
                            text=code_context,
                        )
                    elif tag.kind == "ref":
                        source = (
                            f"{current_class}.{current_function}"
                            if current_class and current_function
                            else (
                                f"{rel_path}:{current_function}"
                                if current_function
                                else rel_path
                            )
                        )
                        references[tag.name].append(
                            (
                                source,
                                tag.line,
                                tag.end_line,
                                tag.type,
                                rel_path,
                                current_class,
                            )
                        )

                # Add a node for the entire file
                G.add_node(
                    rel_path,
                    file=rel_path,
                    type="file",
                    text=file_content,
                )

        for ident, refs in references.items():
            if ident in defines:
                if len(defines[ident]) == 1:
                    target, def_line, end_def_line, def_type, def_file, def_class = (
                        defines[ident][0]
                    )
                    for (
                        source,
                        ref_line,
                        end_ref_line,
                        ref_type,
                        ref_file,
                        ref_class,
                    ) in refs:
                        G.add_edge(
                            source,
                            target,
                            type=ref_type,
                            ident=ident,
                            ref_line=ref_line,
                            end_ref_line=end_ref_line,
                            def_line=def_line,
                            end_def_line=end_def_line,
                        )
                else:
                    for (
                        source,
                        ref_line,
                        end_ref_line,
                        ref_type,
                        ref_file,
                        ref_class,
                    ) in refs:
                        best_match = None
                        best_match_score = -1
                        for (
                            target,
                            def_line,
                            end_def_line,
                            def_type,
                            def_file,
                            def_class,
                        ) in defines[ident]:
                            if source != target:
                                match_score = 0
                                if ref_file == def_file:
                                    match_score += 2
                                elif os.path.dirname(ref_file) == os.path.dirname(
                                    def_file
                                ):
                                    match_score += 1
                                if ref_class == def_class:
                                    match_score += 1
                                if match_score > best_match_score:
                                    best_match = (
                                        target,
                                        def_line,
                                        end_def_line,
                                        def_type,
                                    )
                                    best_match_score = match_score

                        if best_match:
                            target, def_line, end_def_line, def_type = best_match
                            G.add_edge(
                                source,
                                target,
                                type=ref_type,
                                ident=ident,
                                ref_line=ref_line,
                                end_ref_line=end_ref_line,
                                def_line=def_line,
                                end_def_line=end_def_line,
                            )

        end_time = time.time()
        logging.info(f"Parsing completed, time taken: {end_time - start_time} seconds")
        return G

    def is_text_file(self, file_path):
        # Simple check to determine if a file is likely to be a text file
        # You might want to expand this based on your specific needs
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                f.read(1024)
            return True
        except UnicodeDecodeError:
            return False

    @staticmethod
    def get_language_for_file(file_path):
        # Map file extensions to tree-sitter languages
        extension = os.path.splitext(file_path)[1].lower()
        language_map = {
            ".py": get_language("python"),
            ".js": get_language("javascript"),
            ".ts": get_language("typescript"),
            ".c": get_language("c"),
            ".cs": get_language("c_sharp"),
            ".cpp": get_language("cpp"),
            ".el": get_language("elisp"),
            ".ex": get_language("elixir"),
            ".exs": get_language("elixir"),
            ".elm": get_language("elm"),
            ".go": get_language("go"),
            ".java": get_language("java"),
            ".ml": get_language("ocaml"),
            ".mli": get_language("ocaml"),
            ".php": get_language("php"),
            ".ql": get_language("ql"),
            ".rb": get_language("ruby"),
            ".rs": get_language("rust"),
        }
        return language_map.get(extension)

    @staticmethod
    def find_node_by_range(root_node, start_line, node_type):
        def traverse(node):
            if node.start_point[0] <= start_line and node.end_point[0] >= start_line:
                if node_type == "function" and node.type == "function_definition":
                    return node
                elif node_type == "class" and node.type == "class_definition":
                    return node
                for child in node.children:
                    result = traverse(child)
                    if result:
                        return result
            return None

        return traverse(root_node)

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
        return Path(os.path.dirname(__file__)).joinpath(
            "queries", f"tree-sitter-{lang}-tags.scm"
        )
    except KeyError:
        return
