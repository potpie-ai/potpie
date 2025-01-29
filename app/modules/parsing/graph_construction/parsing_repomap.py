import logging
import math
import os
import warnings
from collections import Counter, defaultdict, namedtuple
from pathlib import Path

import networkx as nx
from grep_ast import TreeContext, filename_to_lang
from pygments.lexers import guess_lexer_for_filename
from pygments.token import Token
from pygments.util import ClassNotFound
from tqdm import tqdm
from tree_sitter_languages import get_language, get_parser

from app.core.database import get_db
from app.modules.parsing.graph_construction.parsing_helper import (  # noqa: E402
    ParseHelper,
)

# tree_sitter is throwing a FutureWarning
warnings.simplefilter("ignore", category=FutureWarning)
Tag = namedtuple("Tag", "rel_fname fname line end_line name kind type".split())


class RepoMap:
    # Parsing logic adapted from aider (https://github.com/paul-gauthier/aider)
    # Modified and customized for potpie's parsing needs with detailed tags, relationship tracking etc

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
        self.parse_helper = ParseHelper(next(get_db()))

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
            node_text = node.text.decode("utf-8")

            if tag.startswith("name.definition."):
                kind = "def"
                type = tag.split(".")[-1]

            elif tag.startswith("name.reference."):
                kind = "ref"
                type = tag.split(".")[-1]

            else:
                continue

            saw.add(kind)

            # Enhanced node text extraction for Java methods
            if lang == "java" and type == "method":
                # Handle method calls with object references (e.g., productService.listAllProducts())
                parent = node.parent
                if parent and parent.type == "method_invocation":
                    object_node = parent.child_by_field_name("object")
                    if object_node:
                        node_text = f"{object_node.text.decode('utf-8')}.{node_text}"

            result = Tag(
                rel_fname=rel_fname,
                fname=fname,
                name=node_text,
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

    def create_relationship(
        G, source, target, relationship_type, seen_relationships, extra_data=None
    ):
        """Helper to create relationships with proper direction checking"""
        if source == target:
            return False

        # Determine correct direction based on node types
        source_data = G.nodes[source]
        target_data = G.nodes[target]

        # Prevent duplicate bidirectional relationships
        rel_key = (source, target, relationship_type)
        reverse_key = (target, source, relationship_type)

        if rel_key in seen_relationships or reverse_key in seen_relationships:
            return False

        # Only create relationship if we have right direction:
        # 1. Interface method implementations should point to interface declaration
        # 2. Method calls should point to method definitions
        # 3. Class references should point to class definitions
        valid_direction = False

        if relationship_type == "REFERENCES":
            # Implementation -> Interface
            if (
                source_data.get("type") == "FUNCTION"
                and target_data.get("type") == "FUNCTION"
                and "Impl" in source
            ):  # Implementation class
                valid_direction = True

            # Caller -> Callee
            elif source_data.get("type") == "FUNCTION":
                valid_direction = True

            # Class Usage -> Class Definition
            elif target_data.get("type") == "CLASS":
                valid_direction = True

        if valid_direction:
            G.add_edge(source, target, type=relationship_type, **(extra_data or {}))
            seen_relationships.add(rel_key)
            return True

        return False

    def create_graph(self, repo_dir):
        G = nx.MultiDiGraph()
        defines = defaultdict(set)
        references = defaultdict(set)
        seen_relationships = set()

        for root, dirs, files in os.walk(repo_dir):
            if any(part.startswith(".") for part in root.split(os.sep)):
                continue

            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, repo_dir)

                if not self.parse_helper.is_text_file(file_path):
                    continue

                logging.info(f"\nProcessing file: {rel_path}")

                # Add file node
                file_node_name = rel_path
                if not G.has_node(file_node_name):
                    G.add_node(
                        file_node_name,
                        file=rel_path,
                        type="FILE",
                        text=self.io.read_text(file_path) or "",
                        line=0,
                        end_line=0,
                        name=rel_path.split("/")[-1],
                    )

                current_class = None
                current_method = None

                # Process all tags in file
                for tag in self.get_tags(file_path, rel_path):
                    if tag.kind == "def":
                        if tag.type == "class":
                            node_type = "CLASS"
                            current_class = tag.name
                            current_method = None
                        elif tag.type == "interface":
                            node_type = "INTERFACE"
                            current_class = tag.name
                            current_method = None
                        elif tag.type in ["method", "function"]:
                            node_type = "FUNCTION"
                            current_method = tag.name
                        else:
                            continue

                        # Create fully qualified node name
                        if current_class:
                            node_name = f"{rel_path}:{current_class}.{tag.name}"
                        else:
                            node_name = f"{rel_path}:{tag.name}"

                        # Add node
                        if not G.has_node(node_name):
                            G.add_node(
                                node_name,
                                file=rel_path,
                                line=tag.line,
                                end_line=tag.end_line,
                                type=node_type,
                                name=tag.name,
                                class_name=current_class,
                            )

                            # Add CONTAINS relationship from file
                            rel_key = (file_node_name, node_name, "CONTAINS")
                            if rel_key not in seen_relationships:
                                G.add_edge(
                                    file_node_name,
                                    node_name,
                                    type="CONTAINS",
                                    ident=tag.name,
                                )
                                seen_relationships.add(rel_key)

                        # Record definition
                        defines[tag.name].add(node_name)

                    elif tag.kind == "ref":
                        # Handle references
                        if current_class and current_method:
                            source = f"{rel_path}:{current_class}.{current_method}"
                        elif current_method:
                            source = f"{rel_path}:{current_method}"
                        else:
                            source = rel_path

                        references[tag.name].add(
                            (
                                source,
                                tag.line,
                                tag.end_line,
                                current_class,
                                current_method,
                            )
                        )

        for ident, refs in references.items():
            target_nodes = defines.get(ident, set())

            for source, line, end_line, src_class, src_method in refs:
                for target in target_nodes:
                    if source == target:
                        continue

                    if G.has_node(source) and G.has_node(target):
                        RepoMap.create_relationship(
                            G,
                            source,
                            target,
                            "REFERENCES",
                            seen_relationships,
                            {
                                "ident": ident,
                                "ref_line": line,
                                "end_ref_line": end_line,
                            },
                        )

        return G

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
                if node_type == "FUNCTION" and node.type in [
                    "function_definition",
                    "method",
                    "method_declaration",
                    "function",
                ]:
                    return node
                elif node_type in ["CLASS", "INTERFACE"] and node.type in [
                    "class_definition",
                    "interface",
                    "class",
                    "class_declaration",
                    "interface_declaration",
                ]:
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
