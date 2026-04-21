import os
from collections import defaultdict, namedtuple
from pathlib import Path

import networkx as nx
from grep_ast import filename_to_lang
from loguru import logger
from pygments.lexers import guess_lexer_for_filename
from pygments.token import Token
from pygments.util import ClassNotFound
from tree_sitter import Query, QueryCursor
from tree_sitter_language_pack import get_language, get_parser

Tag = namedtuple("Tag", "rel_fname fname line end_line name kind type".split())


def create_graph(repo_dir):
    G = nx.MultiDiGraph()
    defines = defaultdict(set)
    references = defaultdict(set)
    seen_relationships = set()

    for root, dirs, files in os.walk(repo_dir):
        # Get relative path from repo_dir to avoid skipping paths that contain .repos_local etc.
        try:
            rel_path = Path(root).relative_to(repo_dir)
            # Handle root directory (rel_path == '.') and convert to tuple
            rel_parts = rel_path.parts if rel_path != Path(".") else ()
        except ValueError:
            # If relative_to fails, skip this path (shouldn't happen in normal os.walk)
            continue

        # Skip .git directory (worktrees have .git as a file, not a directory)
        skip_this_dir = False
        if ".git" in rel_parts:
            # Find where .git appears in the relative path
            for i, part in enumerate(rel_parts):
                if part == ".git":
                    # Check if this .git is a directory
                    git_path = Path(repo_dir) / Path(*rel_parts[: i + 1])
                    if git_path.is_dir():
                        # Skip this .git directory
                        skip_this_dir = True
                        break
                    # If it's a file, it's a worktree - continue processing
                    break

        if skip_this_dir:
            continue

        # Skip hidden directories except .github, .vscode, etc. that might contain code
        # Only check relative path parts, not the base path
        if any(
            part.startswith(".") and part not in [".github", ".vscode"]
            for part in rel_parts
        ):
            continue

        for file in files:
            file_path = os.path.join(root, file)
            file_rel_path = os.path.relpath(file_path, repo_dir)

            if not is_text_file(file_path):
                continue

            logger.info(f"\nProcessing file: {file_rel_path}")

            # Add file node
            file_node_name = file_rel_path
            if not G.has_node(file_node_name):
                G.add_node(
                    file_node_name,
                    file=file_rel_path,
                    type="FILE",
                    text=read_text(file_path) or "",
                    line=0,
                    end_line=0,
                    name=file_rel_path.split("/")[-1],
                )

            current_class = None
            current_method = None

            # Process all tags in file
            for tag in get_tags(file_path, file_rel_path):
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
                        node_name = f"{file_rel_path}:{current_class}.{tag.name}"
                    else:
                        node_name = f"{file_rel_path}:{tag.name}"

                    # Add node
                    if not G.has_node(node_name):
                        G.add_node(
                            node_name,
                            file=file_rel_path,
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
                        source = f"{file_rel_path}:{current_class}.{current_method}"
                    elif current_method:
                        source = f"{file_rel_path}:{current_method}"
                    else:
                        source = file_rel_path

                    references[tag.name].add(
                        (
                            source,
                            tag.line,
                            tag.end_line,
                            current_class,
                            current_method,
                        )
                    )

    print(f"DEBUG: references count={len(references)}, defines count={len(defines)}")
    if references:
        print(f"DEBUG: sample ref idents: {list(references.keys())[:5]}")
    if defines:
        print(f"DEBUG: sample define idents: {list(defines.keys())[:5]}")

    for ident, refs in references.items():
        target_nodes = defines.get(ident, set())

        for source, line, end_line, src_class, src_method in refs:
            for target in target_nodes:
                if source == target:
                    continue

                if not G.has_node(source):
                    print(f"MISSING source node: {source}")
                    continue
                if not G.has_node(target):
                    print(f"MISSING target node: {target}")
                    continue

                create_relationship(
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

        # Module-level call -> Function
        elif (
            source_data.get("type") == "FILE" and target_data.get("type") == "FUNCTION"
        ):
            valid_direction = True

    if valid_direction:
        G.add_edge(source, target, type=relationship_type, **(extra_data or {}))
        seen_relationships.add(rel_key)
        return True

    return False


def is_text_file(file_path):
    def open_text_file(file_path):
        """
        Try multiple encodings to detect if file is text.

        Order of encodings to try:
        1. utf-8 (most common)
        2. utf-8-sig (UTF-8 with BOM)
        3. utf-16 (common in Windows C# files)
        4. latin-1/iso-8859-1 (fallback, accepts all byte sequences)
        """
        encodings = ["utf-8", "utf-8-sig", "utf-16", "latin-1"]

        for encoding in encodings:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    # Read first 8KB to detect encoding
                    f.read(8192)
                return True
            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception:
                # Handle other errors (permissions, file not found, etc.)
                return False

        # If all encodings fail, likely a binary file
        return False

    ext = file_path.split(".")[-1]
    exclude_extensions = [
        "png",
        "jpg",
        "jpeg",
        "gif",
        "bmp",
        "tiff",
        "webp",
        "ico",
        "svg",
        "mp4",
        "avi",
        "mov",
        "wmv",
        "flv",
        "ipynb",
    ]
    include_extensions = [
        "py",
        "js",
        "ts",
        "c",
        "cs",
        "cpp",
        "h",
        "hpp",
        "el",
        "ex",
        "exs",
        "elm",
        "go",
        "java",
        "ml",
        "mli",
        "php",
        "ql",
        "rb",
        "rs",
        "md",
        "txt",
        "json",
        "yaml",
        "yml",
        "toml",
        "ini",
        "cfg",
        "conf",
        "xml",
        "html",
        "css",
        "sh",
        "ps1",
        "psm1",
        "md",
        "mdx",
        "xsq",
        "proto",
    ]
    if ext in exclude_extensions:
        return False
    elif ext in include_extensions or open_text_file(file_path):
        return True
    else:
        return False


def read_text(fname):
    """
    Read file with multiple encoding fallbacks.

    Tries encodings in order:
    1. utf-8 (most common)
    2. utf-8-sig (UTF-8 with BOM)
    3. utf-16 (common in Windows files)
    4. latin-1 (fallback that accepts all bytes)
    """
    encodings = ["utf-8", "utf-8-sig", "utf-16", "latin-1"]

    for encoding in encodings:
        try:
            with open(fname, "r", encoding=encoding) as f:
                content = f.read()
                if encoding != "utf-8":
                    logger.info(f"Read {fname} using {encoding} encoding")
                return content
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception:
            logger.exception(f"Error reading {fname}")
            return ""

    logger.warning(
        f"Could not read {fname} with any supported encoding. Skipping this file."
    )
    return ""


def get_tags(fname, rel_fname):
    return list(get_tags_raw(fname, rel_fname))


def get_tags_raw(fname, rel_fname):
    lang = filename_to_lang(fname)
    if not lang:
        logger.debug(f"No lang for {fname}")
        return

    language = get_language(lang)
    parser = get_parser(lang)

    query_scm = get_scm_fname(lang)
    if not query_scm.exists():
        return
    query_scm = query_scm.read_text()

    code = read_text(fname)
    if not code:
        return
    tree = parser.parse(bytes(code, "utf-8"))

    # Run the tags queries
    try:
        query = Query(language, query_scm)
        cursor = QueryCursor(query)
    except Exception as e:
        logger.warning(f"Failed to create query for {fname}: {e}")
        return

    captures = []
    try:
        for _, capture_dict in cursor.matches(tree.root_node):
            for capture_name, nodes in capture_dict.items():
                for node in nodes:
                    captures.append((node, capture_name))
    except Exception as e:
        logger.warning(f"Failed to execute query matches for {fname}: {e}")
        return

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


def get_scm_fname(lang):
    parent_folder = Path(__file__).parent
    query_file = parent_folder / "queries" / f"tree-sitter-{lang}-tags.scm"
    return query_file
