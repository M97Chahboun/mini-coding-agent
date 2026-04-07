import argparse
import json
import re
import shutil
import subprocess
import sys
import textwrap
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path


DOC_NAMES = ("AGENTS.md", "README.md", "pyproject.toml", "package.json")
HELP_TEXT = "/help, /memory, /session, /reset, /exit"
WELCOME_ART = (
    "/\\     /\\\\",
    "{  `---'  }",
    "{  O   O  }",
    "~~>  V  <~~",
    "\\\\  \\|/  /",
    "`-----'__",
)
HELP_DETAILS = textwrap.dedent(
    """\
    Commands:
    /help    Show this help message.
    /memory  Show the agent's distilled working memory.
    /session Show the path to the saved session file.
    /reset   Clear the current session history and memory.
    /exit    Exit the agent.
    """
).strip()
MAX_TOOL_OUTPUT = 4000
MAX_HISTORY = 12000
IGNORED_PATH_NAMES = {".git", ".mini-coding-agent", "__pycache__", ".pytest_cache", ".ruff_cache", ".venv", "venv"}

##############################
#### Six Agent Components ####
##############################
# 1) Live Repo Context -> WorkspaceContext
# 2) Prompt Shape And Cache Reuse -> build_prefix, memory_text, prompt
# 3) Structured Tools, Validation, And Permissions -> build_tools, run_tool, validate_tool, approve, parse, path, tool_*
# 4) Context Reduction And Output Management -> clip, history_text
# 5) Transcripts, Memory, And Resumption -> SessionStore, record, note_tool, ask, reset
# 6) Delegation And Bounded Subagents -> tool_delegate


def now():
    return datetime.now(timezone.utc).isoformat()


# Supporting helper for component 4 (context reduction and output management).
def clip(text, limit=MAX_TOOL_OUTPUT):
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"


def middle(text, limit):
    text = str(text).replace("\n", " ")
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    left = (limit - 3) // 2
    right = limit - 3 - left
    return text[:left] + "..." + text[-right:]


##############################
#### Code Parser Index ######
##############################
class CodeParser:
    """Lightweight code structure extractor for token-efficient indexing.
    
    Extracts class/method/function definitions with line numbers from source files.
    This allows the agent to understand code structure without reading entire files,
    reducing token consumption by ~97% for large files.
    
    Supported languages: Python (.py), with extensible design for more languages.
    """
    
    SUPPORTED_EXTENSIONS = {".py"}
    
    @classmethod
    def parse_file(cls, filepath):
        """Parse a single source file and extract structure information.
        
        Returns a dict with:
        - file: relative path
        - language: detected language
        - classes: list of {name, kind, line_start, line_end, methods: [{name, line_start, line_end}]}
        - functions: list of {name, line_start, line_end} (top-level functions)
        """
        filepath = Path(filepath)
        ext = filepath.suffix.lower()
        
        if ext not in cls.SUPPORTED_EXTENSIONS:
            return None
        
        try:
            content = filepath.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return None
        
        lines = content.splitlines()
        
        if ext == ".py":
            return cls._parse_python(filepath, lines)
        
        return None
    
    @classmethod
    def _parse_python(cls, filepath, lines):
        """Parse Python file using regex-based extraction."""
        result = {
            "file": str(filepath),
            "language": "python",
            "classes": [],
            "functions": [],
        }
        
        # Track indentation levels for class scope detection
        class_stack = []  # Stack of (class_name, indent_level, start_line, methods)
        top_level_indent = None
        
        # Regex patterns
        class_pattern = re.compile(r'^(\s*)class\s+(\w+)(?:\s*\([^)]*\))?\s*:')
        func_pattern = re.compile(r'^(\s*)(?:async\s+)?def\s+(\w+)\s*\([^)]*\)\s*(?:->[^:]+)?:')
        async_func_pattern = re.compile(r'^(\s*)async\s+def\s+(\w+)\s*\([^)]*\)\s*(?:->[^:]+)?:')
        
        current_class = None
        current_methods = []
        class_indent = 0
        
        for line_num, line in enumerate(lines, start=1):
            # Skip empty lines and comments
            stripped = line.lstrip()
            if not stripped or stripped.startswith('#'):
                continue
            
            # Calculate indentation
            indent = len(line) - len(stripped)
            
            # Check for class definition
            class_match = class_pattern.match(line)
            if class_match:
                # Save previous class if any
                if current_class is not None:
                    result["classes"].append({
                        "name": current_class,
                        "kind": "class",
                        "line_start": class_start_line,
                        "line_end": line_num - 1,
                        "methods": current_methods,
                    })
                
                # Start new class
                current_class = class_match.group(2)
                class_indent = indent
                class_start_line = line_num
                current_methods = []
                continue
            
            # Check for method definition inside a class
            if current_class is not None:
                func_match = func_pattern.match(line) or async_func_pattern.match(line)
                if func_match:
                    func_indent = len(func_match.group(1))
                    # Method must be indented more than class
                    if func_indent > class_indent:
                        func_name = func_match.group(2)
                        current_methods.append({
                            "name": func_name,
                            "line_start": line_num,
                            "line_end": line_num,  # Will be updated when we see next item
                        })
                    else:
                        # Exited class scope
                        result["classes"].append({
                            "name": current_class,
                            "kind": "class",
                            "line_start": class_start_line,
                            "line_end": line_num - 1,
                            "methods": current_methods,
                        })
                        current_class = None
                        current_methods = []
                        
                        # Check if this is a top-level function
                        if func_indent == 0:
                            func_name = func_match.group(2)
                            result["functions"].append({
                                "name": func_name,
                                "line_start": line_num,
                                "line_end": line_num,
                            })
                elif indent <= class_indent and stripped and not stripped.startswith('#'):
                    # Exited class scope without seeing a method
                    result["classes"].append({
                        "name": current_class,
                        "kind": "class",
                        "line_start": class_start_line,
                        "line_end": line_num - 1,
                        "methods": current_methods,
                    })
                    current_class = None
                    current_methods = []
            else:
                # Check for top-level function
                func_match = func_pattern.match(line) or async_func_pattern.match(line)
                if func_match:
                    func_indent = len(func_match.group(1))
                    if func_indent == 0:
                        func_name = func_match.group(2)
                        result["functions"].append({
                            "name": func_name,
                            "line_start": line_num,
                            "line_end": line_num,
                        })
        
        # Close final class if any
        if current_class is not None:
            result["classes"].append({
                "name": current_class,
                "kind": "class",
                "line_start": class_start_line,
                "line_end": len(lines),
                "methods": current_methods,
            })
        
        # Update method end lines
        for cls_info in result["classes"]:
            methods = cls_info["methods"]
            for i, method in enumerate(methods):
                if i < len(methods) - 1:
                    method["line_end"] = methods[i + 1]["line_start"] - 1
                else:
                    method["line_end"] = cls_info["line_end"]
        
        return result
    
    @classmethod
    def parse_directory(cls, directory, extensions=None):
        """Parse all supported files in a directory recursively.
        
        Args:
            directory: Path to directory to scan
            extensions: Set of file extensions to include (default: SUPPORTED_EXTENSIONS)
        
        Returns:
            List of parse results for each file
        """
        if extensions is None:
            extensions = cls.SUPPORTED_EXTENSIONS
        
        results = []
        directory = Path(directory)
        
        for ext in extensions:
            for filepath in directory.rglob(f"*{ext}"):
                # Skip ignored paths
                if any(part in IGNORED_PATH_NAMES for part in filepath.relative_to(directory).parts):
                    continue
                
                parsed = cls.parse_file(filepath)
                if parsed:
                    results.append(parsed)
        
        return results
    
    @classmethod
    def to_index_summary(cls, parse_results):
        """Convert parse results to a compact index summary for LLM context.
        
        This creates a token-efficient representation showing only:
        - File paths
        - Class names with method counts
        - Function names
        
        Example output:
        ```
        src/main.py:
          classes: UserService (5 methods), AuthHandler (3 methods)
          functions: main, setup_logging
        
        src/utils.py:
          functions: parse_config, validate_input
        ```
        """
        if not parse_results:
            return "(no code structures found)"
        
        lines = []
        for result in parse_results:
            filepath = result["file"]
            classes = result.get("classes", [])
            functions = result.get("functions", [])
            
            if not classes and not functions:
                continue
            
            lines.append(f"{filepath}:")
            
            if classes:
                class_parts = []
                for cls in classes:
                    method_count = len(cls.get("methods", []))
                    class_parts.append(f"{cls['name']} ({method_count} methods)")
                lines.append(f"  classes: {', '.join(class_parts)}")
            
            if functions:
                func_names = [f["name"] for f in functions]
                lines.append(f"  functions: {', '.join(func_names)}")
        
        return "\n".join(lines) if lines else "(no code structures found)"
    
    @classmethod
    def find_symbol(cls, parse_results, symbol_name):
        """Find a specific class, method, or function by name in parse results.
        
        Args:
            parse_results: List of parse results from parse_file or parse_directory
            symbol_name: Name of the symbol to find (e.g., "UserService" or "createUser")
        
        Returns:
            Dict with file path, symbol info, and line range, or None if not found
        
        Example:
            {
                "file": "src/services/user.py",
                "kind": "method",
                "parent": "UserService",
                "name": "createUser",
                "line_start": 27,
                "line_end": 32
            }
        """
        if not parse_results:
            return None
        
        for result in parse_results:
            filepath = result["file"]
            
            # Search in classes
            for cls in result.get("classes", []):
                if cls["name"] == symbol_name:
                    return {
                        "file": filepath,
                        "kind": "class",
                        "name": cls["name"],
                        "line_start": cls["line_start"],
                        "line_end": cls["line_end"],
                    }
                
                # Search in methods
                for method in cls.get("methods", []):
                    if method["name"] == symbol_name:
                        return {
                            "file": filepath,
                            "kind": "method",
                            "parent": cls["name"],
                            "name": method["name"],
                            "line_start": method["line_start"],
                            "line_end": method["line_end"],
                        }
            
            # Search in top-level functions
            for func in result.get("functions", []):
                if func["name"] == symbol_name:
                    return {
                        "file": filepath,
                        "kind": "function",
                        "name": func["name"],
                        "line_start": func["line_start"],
                        "line_end": func["line_end"],
                    }
        
        return None
    
    @classmethod
    def find_symbols_in_file(cls, filepath, symbol_names):
        """Find multiple symbols in a single file.
        
        Args:
            filepath: Path to the file to parse
            symbol_names: List of symbol names to find
        
        Returns:
            Dict mapping symbol names to their locations, or empty dict if none found
        """
        parse_result = cls.parse_file(filepath)
        if not parse_result:
            return {}
        
        results = {}
        for name in symbol_names:
            found = cls.find_symbol([parse_result], name)
            if found:
                results[name] = found
        
        return results


##############################
#### 1) Live Repo Context ####
##############################
class WorkspaceContext:
    def __init__(self, cwd, repo_root, branch, default_branch, status, recent_commits, project_docs):
        self.cwd = cwd
        self.repo_root = repo_root
        self.branch = branch
        self.default_branch = default_branch
        self.status = status
        self.recent_commits = recent_commits
        self.project_docs = project_docs

    @classmethod
    def build(cls, cwd):
        cwd = Path(cwd).resolve()

        def git(args, fallback=""):
            try:
                result = subprocess.run(
                    ["git", *args],
                    cwd=cwd,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=5,
                )
                return result.stdout.strip() or fallback
            except Exception:
                return fallback

        repo_root = Path(git(["rev-parse", "--show-toplevel"], str(cwd))).resolve()
        docs = {}
        for base in (repo_root, cwd):
            for name in DOC_NAMES:
                path = base / name
                if not path.exists():
                    continue
                key = str(path.relative_to(repo_root))
                if key in docs:
                    continue
                docs[key] = clip(path.read_text(encoding="utf-8", errors="replace"), 1200)

        return cls(
            cwd=str(cwd),
            repo_root=str(repo_root),
            branch=git(["branch", "--show-current"], "-") or "-",
            default_branch=(git(["symbolic-ref", "--short", "refs/remotes/origin/HEAD"], "origin/main") or "origin/main").removeprefix("origin/"),
            status=clip(git(["status", "--short"], "clean") or "clean", 1500),
            recent_commits=[line for line in git(["log", "--oneline", "-5"]).splitlines() if line],
            project_docs=docs,
        )

    def text(self):
        commits = "\n".join(f"- {line}" for line in self.recent_commits) or "- none"
        docs = "\n".join(f"- {path}\n{snippet}" for path, snippet in self.project_docs.items()) or "- none"
        return textwrap.dedent(
            f"""\
            Workspace:
            - cwd: {self.cwd}
            - repo_root: {self.repo_root}
            - branch: {self.branch}
            - default_branch: {self.default_branch}
            - status:
            {self.status}
            - recent_commits:
            {commits}
            - project_docs:
            {docs}
            """
        ).strip()


##############################
#### 5) Session Memory #######
##############################
class SessionStore:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, session_id):
        return self.root / f"{session_id}.json"

    def save(self, session):
        path = self.path(session["id"])
        path.write_text(json.dumps(session, indent=2), encoding="utf-8")
        return path

    def load(self, session_id):
        return json.loads(self.path(session_id).read_text(encoding="utf-8"))

    def latest(self):
        files = sorted(self.root.glob("*.json"), key=lambda path: path.stat().st_mtime)
        return files[-1].stem if files else None


class FakeModelClient:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.prompts = []

    def complete(self, prompt, max_new_tokens):
        self.prompts.append(prompt)
        if not self.outputs:
            raise RuntimeError("fake model ran out of outputs")
        return self.outputs.pop(0)


class OllamaModelClient:
    def __init__(self, model, host, temperature, top_p, timeout, stream=False):
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout
        self.stream = stream

    def complete(self, prompt, max_new_tokens):
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": self.stream,
            "raw": False,
            "think": False,
            "options": {
                "num_predict": max_new_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
            },
        }
        request = urllib.request.Request(
            self.host + "/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                if self.stream:
                    # Stream the response and yield chunks for real-time display
                    return self._stream_response(response)
                else:
                    data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama request failed with HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                "Could not reach Ollama.\n"
                "Make sure `ollama serve` is running and the model is available.\n"
                f"Host: {self.host}\n"
                f"Model: {self.model}"
            ) from exc

        if data.get("error"):
            raise RuntimeError(f"Ollama error: {data['error']}")
        return data.get("response", "")

    def _stream_response(self, response):
        """Generator that streams response chunks for real-time display."""
        chunks = []
        while True:
            chunk = response.read(4096)
            if not chunk:
                break
            for line in chunk.decode("utf-8").splitlines():
                if line.strip():
                    data = json.loads(line)
                    if "response" in data:
                        chunks.append(data["response"])
                        # Yield chunk for streaming display
                        yield data["response"]
                    if data.get("done", False):
                        break
        return "".join(chunks)


class MiniAgent:
    def __init__(
        self,
        model_client,
        workspace,
        session_store,
        session=None,
        approval_policy="ask",
        max_steps=6,
        max_new_tokens=512,
        depth=0,
        max_depth=1,
        read_only=False,
    ):
        self.model_client = model_client
        self.workspace = workspace
        self.root = Path(workspace.repo_root)
        self.session_store = session_store
        self.approval_policy = approval_policy
        self.max_steps = max_steps
        self.max_new_tokens = max_new_tokens
        self.depth = depth
        self.max_depth = max_depth
        self.read_only = read_only
        self.session = session or {
            "id": datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6],
            "created_at": now(),
            "workspace_root": workspace.repo_root,
            "history": [],
            "memory": {"task": "", "files": [], "notes": []},
        }
        self.tools = self.build_tools()
        self.prefix = self.build_prefix()
        self.session_path = self.session_store.save(self.session)

    @classmethod
    def from_session(cls, model_client, workspace, session_store, session_id, **kwargs):
        return cls(
            model_client=model_client,
            workspace=workspace,
            session_store=session_store,
            session=session_store.load(session_id),
            **kwargs,
        )

    @staticmethod
    def remember(bucket, item, limit):
        if not item:
            return
        if item in bucket:
            bucket.remove(item)
        bucket.append(item)
        del bucket[:-limit]

    ###############################################
    #### 3) Structured Tools And Permissions ######
    ###############################################
    def build_tools(self):
        tools = {
            "list_files": {
                "schema": {"path": "str='.'"},
                "risky": False,
                "description": "List files in the workspace.",
                "run": self.tool_list_files,
            },
            "read_file": {
                "schema": {"path": "str", "start": "int=1", "end": "int=200"},
                "risky": False,
                "description": "Read a UTF-8 file by line range.",
                "run": self.tool_read_file,
            },
            "read_lines": {
                "schema": {"path": "str", "start": "int", "end": "int"},
                "risky": False,
                "description": "Read specific lines from a source file (use with code index).",
                "run": self.tool_read_lines,
            },
            "code_index": {
                "schema": {"path": "str='.'"},
                "risky": False,
                "description": "Generate a token-efficient code structure index for a directory.",
                "run": self.tool_code_index,
            },
            "find_symbol": {
                "schema": {"symbol": "str", "path": "str='.'"},
                "risky": False,
                "description": "Find a class/method/function by name and return its exact line range for surgical reading.",
                "run": self.tool_find_symbol,
            },
            "search": {
                "schema": {"pattern": "str", "path": "str='.'"},
                "risky": False,
                "description": "Search the workspace with rg or a simple fallback.",
                "run": self.tool_search,
            },
            "run_shell": {
                "schema": {"command": "str", "timeout": "int=20"},
                "risky": True,
                "description": "Run a shell command in the repo root.",
                "run": self.tool_run_shell,
            },
            "write_file": {
                "schema": {"path": "str", "content": "str"},
                "risky": True,
                "description": "Write a text file.",
                "run": self.tool_write_file,
            },
            "patch_file": {
                "schema": {"path": "str", "old_text": "str", "new_text": "str"},
                "risky": True,
                "description": "Replace one exact text block in a file.",
                "run": self.tool_patch_file,
            },
            "write_files": {
                "schema": {"files": "list"},
                "risky": True,
                "description": "Atomically write multiple files; all succeed or none (rollback on failure). Each file: {path, content}.",
                "run": self.tool_write_files,
            },
            "patch_files": {
                "schema": {"patches": "list"},
                "risky": True,
                "description": "Atomically patch multiple files; all succeed or none (rollback on failure). Each patch: {path, old_text, new_text}.",
                "run": self.tool_patch_files,
            },
        }
        if self.depth < self.max_depth:
            tools["delegate"] = {
                "schema": {"task": "str", "max_steps": "int=3"},
                "risky": False,
                "description": "Ask a bounded read-only child agent to investigate.",
                "run": self.tool_delegate,
            }
        return tools

    ############################################
    #### 2) Prompt Shape And Cache Reuse #######
    ############################################
    def build_prefix(self):
        tool_lines = []
        for name, tool in self.tools.items():
            fields = ", ".join(f"{key}: {value}" for key, value in tool["schema"].items())
            risk = "approval required" if tool["risky"] else "safe"
            tool_lines.append(f"- {name}({fields}) [{risk}] {tool['description']}")
        tool_text = "\n".join(tool_lines)
        examples = "\n".join(
            [
                '<tool>{"name":"list_files","args":{"path":"."}}</tool>',
                '<tool>{"name":"read_file","args":{"path":"README.md","start":1,"end":80}}</tool>',
                '<tool name="write_file" path="binary_search.py"><content>def binary_search(nums, target):\n    return -1\n</content></tool>',
                '<tool name="patch_file" path="binary_search.py"><old_text>return -1</old_text><new_text>return mid</new_text></tool>',
                '<tool name="write_files"><files>[{"path": "a.py", "content": "print(1)"}, {"path": "b.py", "content": "print(2)"}]</files></tool>',
                '<tool name="patch_files"><patches>[{"path": "main.py", "old_text": "x=1", "new_text": "x=2"}]</patches></tool>',
                '<tool>{"name":"run_shell","args":{"command":"uv run --with pytest python -m pytest -q","timeout":20}}</tool>',
                '<tool>{"name":"code_index","args":{"path":"."}}</tool>',
                '<tool>{"name":"find_symbol","args":{"symbol":"UserService","path":"src"}}</tool>',
                '<tool>{"name":"read_lines","args":{"path":"main.py","start":27,"end":32}}</tool>',
                "<final>Done.</final>",
            ]
        )
        return textwrap.dedent(
            f"""\
            You are Mini-Coding-Agent, a small local coding agent running through Ollama.

            Rules:
            - Use tools instead of guessing about the workspace.
            - Return exactly one <tool>...</tool> or one <final>...</final>.
            - Tool calls must look like:
              <tool>{{"name":"tool_name","args":{{...}}}}</tool>
            - For write_file and patch_file with multi-line text, prefer XML style:
              <tool name="write_file" path="file.py"><content>...</content></tool>
            - Final answers must look like:
              <final>your answer</final>
            - Never invent tool results.
            - Keep answers concise and concrete.
            - If the user asks you to create or update a specific file and the path is clear, use write_file or patch_file instead of repeatedly listing files.
            - Before writing tests for existing code, read the implementation first.
            - When writing tests, match the current implementation unless the user explicitly asked you to change the code.
            - New files should be complete and runnable, including obvious imports.
            - Do not repeat the same tool call with the same arguments if it did not help. Choose a different tool or return a final answer.
            - Required tool arguments must not be empty. Do not call read_file, write_file, patch_file, run_shell, delegate, write_files, or patch_files with args={{}}.
            - For token efficiency: use code_index first to get an overview, then find_symbol to locate specific classes/methods, then read_lines for surgical reading of only the relevant code.

            Tools:
            {tool_text}

            Valid response examples:
            {examples}

            {self.workspace.text()}
            """
        ).strip()

    def memory_text(self):
        memory = self.session["memory"]
        return textwrap.dedent(
            f"""\
            Memory:
            - task: {memory['task'] or "-"}
            - files: {", ".join(memory["files"]) or "-"}
            - notes:
              {chr(10).join(f"- {note}" for note in memory["notes"]) or "- none"}
            """
        ).strip()

    #####################################################
    #### 4) Context Reduction And Output Management #####
    #####################################################
    def history_text(self):
        history = self.session["history"]
        if not history:
            return "- empty"

        lines = []
        seen_reads = set()
        recent_start = max(0, len(history) - 6)
        for index, item in enumerate(history):
            recent = index >= recent_start
            if item["role"] == "tool" and item["name"] in ("write_file", "patch_file"):
                path = str(item["args"].get("path", ""))
                seen_reads.discard(path)
            if item["role"] == "tool" and item["name"] == "read_file" and not recent:
                path = str(item["args"].get("path", ""))
                if path in seen_reads:
                    continue
                seen_reads.add(path)

            if item["role"] == "tool":
                limit = 900 if recent else 180
                lines.append(f"[tool:{item['name']}] {json.dumps(item['args'], sort_keys=True)}")
                lines.append(clip(item["content"], limit))
            else:
                limit = 900 if recent else 220
                lines.append(f"[{item['role']}] {clip(item['content'], limit)}")

        return clip("\n".join(lines), MAX_HISTORY)

    ########################################################
    #### 2) Prompt Shape And Cache Reuse (Continued) #######
    ########################################################
    def prompt(self, user_message):
        return textwrap.dedent(
            f"""\
            {self.prefix}

            {self.memory_text()}

            Transcript:
            {self.history_text()}

            Current user request:
            {user_message}
            """
        ).strip()

    ###############################################
    #### 5) Session Memory (Continued) ###########
    ###############################################
    def record(self, item):
        self.session["history"].append(item)
        self.session_path = self.session_store.save(self.session)

    def note_tool(self, name, args, result):
        memory = self.session["memory"]
        path = args.get("path")
        if name in {"read_file", "write_file", "patch_file", "write_files", "patch_files"} and path:
            self.remember(memory["files"], str(path), 8)
        # For multi-file operations, track all paths
        if name == "write_files":
            for file_spec in args.get("files", []):
                self.remember(memory["files"], str(file_spec.get("path", "")), 8)
        if name == "patch_files":
            for patch_spec in args.get("patches", []):
                self.remember(memory["files"], str(patch_spec.get("path", "")), 8)
        note = f"{name}: {clip(str(result).replace(chr(10), ' '), 220)}"
        self.remember(memory["notes"], note, 5)

    def ask(self, user_message, stream_output=True):
        memory = self.session["memory"]
        if not memory["task"]:
            memory["task"] = clip(user_message.strip(), 300)
        self.record({"role": "user", "content": user_message, "created_at": now()})

        tool_steps = 0
        attempts = 0
        max_attempts = max(self.max_steps * 3, self.max_steps + 4)

        while tool_steps < self.max_steps and attempts < max_attempts:
            attempts += 1
            raw_response = self.model_client.complete(self.prompt(user_message), self.max_new_tokens)
            
            # Handle streaming response
            if hasattr(raw_response, '__iter__') and not isinstance(raw_response, str):
                # It's a generator (streaming mode)
                if stream_output:
                    print("\n[Assistant streaming]: ", end="", flush=True)
                chunks = []
                for chunk in raw_response:
                    if stream_output:
                        print(chunk, end="", flush=True)
                    chunks.append(chunk)
                raw = "".join(chunks)
                if stream_output:
                    print()  # Newline after streaming
            else:
                raw = raw_response
            
            kind, payload = self.parse(raw)

            if kind == "tool":
                tool_steps += 1
                name = payload.get("name", "")
                args = payload.get("args", {})
                result = self.run_tool(name, args)
                self.record(
                    {
                        "role": "tool",
                        "name": name,
                        "args": args,
                        "content": result,
                        "created_at": now(),
                    }
                )
                self.note_tool(name, args, result)
                continue

            if kind == "retry":
                self.record({"role": "assistant", "content": payload, "created_at": now()})
                continue

            final = (payload or raw).strip()
            self.record({"role": "assistant", "content": final, "created_at": now()})
            self.remember(memory["notes"], clip(final, 220), 5)
            return final

        if attempts >= max_attempts and tool_steps < self.max_steps:
            final = "Stopped after too many malformed model responses without a valid tool call or final answer."
        else:
            final = "Stopped after reaching the step limit without a final answer."
        self.record({"role": "assistant", "content": final, "created_at": now()})
        return final

    #############################################################
    #### 3) Structured Tools, Validation, And Permissions #######
    #############################################################
    def run_tool(self, name, args):
        tool = self.tools.get(name)
        if tool is None:
            return f"error: unknown tool '{name}'"
        try:
            self.validate_tool(name, args)
        except Exception as exc:
            example = self.tool_example(name)
            message = f"error: invalid arguments for {name}: {exc}"
            if example:
                message += f"\nexample: {example}"
            return message
        if self.repeated_tool_call(name, args):
            return f"error: repeated identical tool call for {name}; choose a different tool or return a final answer"
        if tool["risky"] and not self.approve(name, args):
            return f"error: approval denied for {name}"
        try:
            return clip(tool["run"](args))
        except Exception as exc:
            return f"error: tool {name} failed: {exc}"

    def repeated_tool_call(self, name, args):
        tool_events = [item for item in self.session["history"] if item["role"] == "tool"]
        if len(tool_events) < 2:
            return False
        recent = tool_events[-2:]
        return all(item["name"] == name and item["args"] == args for item in recent)

    def tool_example(self, name):
        examples = {
            "list_files": '<tool>{"name":"list_files","args":{"path":"."}}</tool>',
            "read_file": '<tool>{"name":"read_file","args":{"path":"README.md","start":1,"end":80}}</tool>',
            "search": '<tool>{"name":"search","args":{"pattern":"binary_search","path":"."}}</tool>',
            "run_shell": '<tool>{"name":"run_shell","args":{"command":"uv run --with pytest python -m pytest -q","timeout":20}}</tool>',
            "write_file": '<tool name="write_file" path="binary_search.py"><content>def binary_search(nums, target):\n    return -1\n</content></tool>',
            "patch_file": '<tool name="patch_file" path="binary_search.py"><old_text>return -1</old_text><new_text>return mid</new_text></tool>',
            "delegate": '<tool>{"name":"delegate","args":{"task":"inspect README.md","max_steps":3}}</tool>',
        }
        return examples.get(name, "")

    def validate_tool(self, name, args):
        args = args or {}

        if name == "list_files":
            path = self.path(args.get("path", "."))
            if not path.is_dir():
                raise ValueError("path is not a directory")
            return

        if name == "read_file":
            path = self.path(args["path"])
            if not path.is_file():
                raise ValueError("path is not a file")
            start = int(args.get("start", 1))
            end = int(args.get("end", 200))
            if start < 1 or end < start:
                raise ValueError("invalid line range")
            return

        if name == "read_lines":
            path = self.path(args["path"])
            if not path.is_file():
                raise ValueError("path is not a file")
            start = int(args.get("start"))
            end = int(args.get("end"))
            if not start or not end:
                raise ValueError("start and end are required")
            if start < 1 or end < start:
                raise ValueError("invalid line range")
            return

        if name == "code_index":
            path = self.path(args.get("path", "."))
            if not path.is_dir():
                raise ValueError("path is not a directory")
            return

        if name == "search":
            pattern = str(args.get("pattern", "")).strip()
            if not pattern:
                raise ValueError("pattern must not be empty")
            self.path(args.get("path", "."))
            return

        if name == "run_shell":
            command = str(args.get("command", "")).strip()
            if not command:
                raise ValueError("command must not be empty")
            timeout = int(args.get("timeout", 20))
            if timeout < 1 or timeout > 120:
                raise ValueError("timeout must be in [1, 120]")
            return

        if name == "write_file":
            path = self.path(args["path"])
            if path.exists() and path.is_dir():
                raise ValueError("path is a directory")
            if "content" not in args:
                raise ValueError("missing content")
            return

        if name == "patch_file":
            path = self.path(args["path"])
            if not path.is_file():
                raise ValueError("path is not a file")
            old_text = str(args.get("old_text", ""))
            if not old_text:
                raise ValueError("old_text must not be empty")
            if "new_text" not in args:
                raise ValueError("missing new_text")
            text = path.read_text(encoding="utf-8")
            count = text.count(old_text)
            if count != 1:
                raise ValueError(f"old_text must occur exactly once, found {count}")
            return

        if name == "delegate":
            if self.depth >= self.max_depth:
                raise ValueError("delegate depth exceeded")
            task = str(args.get("task", "")).strip()
            if not task:
                raise ValueError("task must not be empty")
            return

        if name == "write_files":
            files = args.get("files")
            if not isinstance(files, list) or len(files) == 0:
                raise ValueError("files must be a non-empty list")
            for idx, file_spec in enumerate(files):
                if not isinstance(file_spec, dict):
                    raise ValueError(f"file {idx} must be an object")
                if "path" not in file_spec:
                    raise ValueError(f"file {idx} missing 'path'")
                if "content" not in file_spec:
                    raise ValueError(f"file {idx} missing 'content'")
                path = self.path(file_spec["path"])
                if path.exists() and path.is_dir():
                    raise ValueError(f"file {idx} path is a directory: {file_spec['path']}")
            return

        if name == "patch_files":
            patches = args.get("patches")
            if not isinstance(patches, list) or len(patches) == 0:
                raise ValueError("patches must be a non-empty list")
            for idx, patch_spec in enumerate(patches):
                if not isinstance(patch_spec, dict):
                    raise ValueError(f"patch {idx} must be an object")
                if "path" not in patch_spec:
                    raise ValueError(f"patch {idx} missing 'path'")
                if "old_text" not in patch_spec:
                    raise ValueError(f"patch {idx} missing 'old_text'")
                if "new_text" not in patch_spec:
                    raise ValueError(f"patch {idx} missing 'new_text'")
                path = self.path(patch_spec["path"])
                if not path.is_file():
                    raise ValueError(f"patch {idx} path is not a file: {patch_spec['path']}")
                old_text = str(patch_spec.get("old_text", ""))
                if not old_text:
                    raise ValueError(f"patch {idx} old_text must not be empty")
                text = path.read_text(encoding="utf-8")
                count = text.count(old_text)
                if count != 1:
                    raise ValueError(f"patch {idx} old_text must occur exactly once in {patch_spec['path']}, found {count}")
            return

    def approve(self, name, args):
        if self.read_only:
            return False
        if self.approval_policy == "auto":
            return True
        if self.approval_policy == "never":
            return False
        try:
            answer = input(f"approve {name} {json.dumps(args, ensure_ascii=True)}? [y/N] ")
        except EOFError:
            return False
        return answer.strip().lower() in {"y", "yes"}

    @staticmethod
    def parse(raw):
        raw = str(raw)
        if "<tool>" in raw and ("<final>" not in raw or raw.find("<tool>") < raw.find("<final>")):
            body = MiniAgent.extract(raw, "tool")
            try:
                payload = json.loads(body)
            except Exception:
                return "retry", MiniAgent.retry_notice("model returned malformed tool JSON")
            if not isinstance(payload, dict):
                return "retry", MiniAgent.retry_notice("tool payload must be a JSON object")
            if not str(payload.get("name", "")).strip():
                return "retry", MiniAgent.retry_notice("tool payload is missing a tool name")
            args = payload.get("args", {})
            if args is None:
                payload["args"] = {}
            elif not isinstance(args, dict):
                return "retry", MiniAgent.retry_notice()
            return "tool", payload
        if "<tool" in raw and ("<final>" not in raw or raw.find("<tool") < raw.find("<final>")):
            payload = MiniAgent.parse_xml_tool(raw)
            if payload is not None:
                return "tool", payload
            return "retry", MiniAgent.retry_notice()
        if "<final>" in raw:
            final = MiniAgent.extract(raw, "final").strip()
            if final:
                return "final", final
            return "retry", MiniAgent.retry_notice("model returned an empty <final> answer")
        raw = raw.strip()
        if raw:
            return "final", raw
        return "retry", MiniAgent.retry_notice("model returned an empty response")

    @staticmethod
    def retry_notice(problem=None):
        prefix = "Runtime notice"
        if problem:
            prefix += f": {problem}"
        else:
            prefix += ": model returned malformed tool output"
        return (
            f"{prefix}. Reply with a valid <tool> call or a non-empty <final> answer. "
            'For multi-line files, prefer <tool name="write_file" path="file.py"><content>...</content></tool>.'
        )

    @staticmethod
    def parse_xml_tool(raw):
        match = re.search(r"<tool(?P<attrs>[^>]*)>(?P<body>.*?)</tool>", raw, re.S)
        if not match:
            return None
        attrs = MiniAgent.parse_attrs(match.group("attrs"))
        name = str(attrs.pop("name", "")).strip()
        if not name:
            return None

        body = match.group("body")
        args = dict(attrs)
        for key in ("content", "old_text", "new_text", "command", "task", "pattern", "path"):
            if f"<{key}>" in body:
                args[key] = MiniAgent.extract_raw(body, key)

        body_text = body.strip("\n")
        if name == "write_file" and "content" not in args and body_text:
            args["content"] = body_text
        if name == "delegate" and "task" not in args and body_text:
            args["task"] = body_text.strip()
        return {"name": name, "args": args}

    @staticmethod
    def parse_attrs(text):
        attrs = {}
        for match in re.finditer(r"""([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:"([^"]*)"|'([^']*)')""", text):
            attrs[match.group(1)] = match.group(2) if match.group(2) is not None else match.group(3)
        return attrs

    @staticmethod
    def extract(text, tag):
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"
        start = text.find(start_tag)
        if start == -1:
            return text
        start += len(start_tag)
        end = text.find(end_tag, start)
        if end == -1:
            return text[start:].strip()
        return text[start:end].strip()

    @staticmethod
    def extract_raw(text, tag):
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"
        start = text.find(start_tag)
        if start == -1:
            return text
        start += len(start_tag)
        end = text.find(end_tag, start)
        if end == -1:
            return text[start:]
        return text[start:end]

    def reset(self):
        self.session["history"] = []
        self.session["memory"] = {"task": "", "files": [], "notes": []}
        self.session_store.save(self.session)

    def path_is_within_root(self, resolved):
        probe = resolved
        while not probe.exists() and probe.parent != probe:
            probe = probe.parent
        for candidate in (probe, *probe.parents):
            try:
                if candidate.samefile(self.root):
                    return True
            except OSError:
                continue
        return False

    def path(self, raw_path):
        path = Path(raw_path)
        path = path if path.is_absolute() else self.root / path
        resolved = path.resolve()
        if not self.path_is_within_root(resolved):
            raise ValueError(f"path escapes workspace: {raw_path}")
        return resolved

    def tool_list_files(self, args):
        path = self.path(args.get("path", "."))
        if not path.is_dir():
            raise ValueError("path is not a directory")
        entries = [
            item for item in sorted(path.iterdir(), key=lambda item: (item.is_file(), item.name.lower()))
            if item.name not in IGNORED_PATH_NAMES
        ]
        lines = []
        for entry in entries[:200]:
            kind = "[D]" if entry.is_dir() else "[F]"
            lines.append(f"{kind} {entry.relative_to(self.root)}")
        return "\n".join(lines) or "(empty)"

    def tool_read_file(self, args):
        path = self.path(args["path"])
        if not path.is_file():
            raise ValueError("path is not a file")
        start = int(args.get("start", 1))
        end = int(args.get("end", 200))
        if start < 1 or end < start:
            raise ValueError("invalid line range")
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        body = "\n".join(f"{number:>4}: {line}" for number, line in enumerate(lines[start - 1:end], start=start))
        return f"# {path.relative_to(self.root)}\n{body}"

    def tool_read_lines(self, args):
        """Read specific lines from a source file (for use with code index)."""
        path = self.path(args["path"])
        if not path.is_file():
            raise ValueError("path is not a file")
        start = int(args.get("start"))
        end = int(args.get("end"))
        if not start or not end:
            raise ValueError("start and end line numbers are required")
        if start < 1 or end < start:
            raise ValueError("invalid line range")
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        body = "\n".join(f"{number:>4}: {line}" for number, line in enumerate(lines[start - 1:end], start=start))
        return f"# {path.relative_to(self.root)}\n{body}"

    def tool_code_index(self, args):
        """Generate a token-efficient code structure index for a directory."""
        path = self.path(args.get("path", "."))
        if not path.is_dir():
            raise ValueError("path is not a directory")
        
        # Parse all supported files in the directory
        parse_results = CodeParser.parse_directory(path)
        
        # Generate compact summary
        summary = CodeParser.to_index_summary(parse_results)
        
        # Also include detailed JSON for programmatic access
        import json
        details = json.dumps(parse_results, indent=2)
        
        return f"Code Structure Index Summary:\n{summary}\n\n--- Full Details ---\n{details}"
    
    def tool_find_symbol(self, args):
        """Find a specific class, method, or function by name and return its exact line range.
        
        Use this after code_index to locate symbols without reading entire files.
        Returns the file path and line range for surgical reading with read_lines.
        """
        symbol_name = str(args.get("symbol", "")).strip()
        if not symbol_name:
            raise ValueError("symbol name is required")
        
        path = self.path(args.get("path", "."))
        if not path.exists():
            raise ValueError("path does not exist")
        
        # Parse the directory or single file
        if path.is_dir():
            parse_results = CodeParser.parse_directory(path)
        else:
            result = CodeParser.parse_file(path)
            parse_results = [result] if result else []
        
        # Find the symbol
        found = CodeParser.find_symbol(parse_results, symbol_name)
        
        if not found:
            return f"Symbol '{symbol_name}' not found in {path}"
        
        # Format result with usage hint
        import json
        return (
            f"Found {found['kind']} '{found['name']}'"
            + (f" in {found['parent']}" if found.get('parent') else "")
            + f":\nFile: {found['file']}\nLines: {found['line_start']}-{found['line_end']}\n\n"
            f"Use read_lines to view: <tool>{{\"name\":\"read_lines\",\"args\":{{\"path\":\"{found['file']}\","
            f"\"start\":{found['line_start']},\"end\":{found['line_end']}}}}}</tool>"
        )

    def tool_search(self, args):
        pattern = str(args.get("pattern", "")).strip()
        if not pattern:
            raise ValueError("pattern must not be empty")
        path = self.path(args.get("path", "."))

        if shutil.which("rg"):
            result = subprocess.run(
                ["rg", "-n", "--smart-case", "--max-count", "200", pattern, str(path)],
                cwd=self.root,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip() or result.stderr.strip() or "(no matches)"

        matches = []
        files = [path] if path.is_file() else [
            item for item in path.rglob("*")
            if item.is_file() and not any(part in IGNORED_PATH_NAMES for part in item.relative_to(self.root).parts)
        ]
        for file_path in files:
            for number, line in enumerate(file_path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
                if pattern.lower() in line.lower():
                    matches.append(f"{file_path.relative_to(self.root)}:{number}:{line}")
                    if len(matches) >= 200:
                        return "\n".join(matches)
        return "\n".join(matches) or "(no matches)"

    def tool_run_shell(self, args):
        command = str(args.get("command", "")).strip()
        if not command:
            raise ValueError("command must not be empty")
        timeout = int(args.get("timeout", 20))
        if timeout < 1 or timeout > 120:
            raise ValueError("timeout must be in [1, 120]")
        result = subprocess.run(
            command,
            cwd=self.root,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return textwrap.dedent(
            f"""\
            exit_code: {result.returncode}
            stdout:
            {result.stdout.strip() or "(empty)"}
            stderr:
            {result.stderr.strip() or "(empty)"}
            """
        ).strip()

    def tool_write_file(self, args):
        path = self.path(args["path"])
        content = str(args["content"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"wrote {path.relative_to(self.root)} ({len(content)} chars)"

    def tool_patch_file(self, args):
        path = self.path(args["path"])
        if not path.is_file():
            raise ValueError("path is not a file")
        old_text = str(args.get("old_text", ""))
        if not old_text:
            raise ValueError("old_text must not be empty")
        if "new_text" not in args:
            raise ValueError("missing new_text")
        text = path.read_text(encoding="utf-8")
        count = text.count(old_text)
        if count != 1:
            raise ValueError(f"old_text must occur exactly once, found {count}")
        path.write_text(text.replace(old_text, str(args["new_text"]), 1), encoding="utf-8")
        return f"patched {path.relative_to(self.root)}"

    def tool_write_files(self, args):
        """Atomically write multiple files with rollback on failure."""
        files = args.get("files", [])
        written_paths = []
        original_contents = {}
        
        try:
            # Phase 1: Write all files, tracking what we've done
            for idx, file_spec in enumerate(files):
                path = self.path(file_spec["path"])
                content = str(file_spec["content"])
                
                # Save original content for rollback if file exists
                if path.exists():
                    original_contents[path] = path.read_text(encoding="utf-8")
                
                # Create parent directories and write
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
                written_paths.append(path)
            
            # Success - return summary
            summaries = [f"wrote {p.relative_to(self.root)}" for p in written_paths]
            return "atomic write successful:\\n" + "\\n".join(summaries)
            
        except Exception as exc:
            # Rollback: restore originals or remove newly created files
            for path in reversed(written_paths):
                try:
                    if path in original_contents:
                        path.write_text(original_contents[path], encoding="utf-8")
                    elif path.exists():
                        path.unlink()
                except OSError:
                    pass  # Best effort rollback
            raise RuntimeError(f"atomic write failed: {exc}; rolled back changes") from exc

    def tool_patch_files(self, args):
        """Atomically patch multiple files with rollback on failure."""
        patches = args.get("patches", [])
        patched_paths = []
        original_contents = {}
        
        try:
            # Phase 1: Read all files and validate patches
            for idx, patch_spec in enumerate(patches):
                path = self.path(patch_spec["path"])
                if not path.is_file():
                    raise ValueError(f"patch {idx} path is not a file: {patch_spec['path']}")
                
                # Save original content
                original_contents[path] = path.read_text(encoding="utf-8")
                
                # Validate patch can be applied
                old_text = str(patch_spec.get("old_text", ""))
                if not old_text:
                    raise ValueError(f"patch {idx} old_text must not be empty")
                text = original_contents[path]
                count = text.count(old_text)
                if count != 1:
                    raise ValueError(f"patch {idx} old_text must occur exactly once in {patch_spec['path']}, found {count}")
            
            # Phase 2: Apply all patches
            for idx, patch_spec in enumerate(patches):
                path = self.path(patch_spec["path"])
                old_text = str(patch_spec["old_text"])
                new_text = str(patch_spec["new_text"])
                text = original_contents[path]
                
                # Apply patch
                new_content = text.replace(old_text, new_text, 1)
                path.write_text(new_content, encoding="utf-8")
                patched_paths.append(path)
            
            # Success - return summary
            summaries = [f"patched {p.relative_to(self.root)}" for p in patched_paths]
            return "atomic patch successful:\\n" + "\\n".join(summaries)
            
        except Exception as exc:
            # Rollback: restore all original contents
            for path in reversed(patched_paths):
                try:
                    if path in original_contents:
                        path.write_text(original_contents[path], encoding="utf-8")
                except OSError:
                    pass  # Best effort rollback
            raise RuntimeError(f"atomic patch failed: {exc}; rolled back changes") from exc

    ###################################################
    #### 6) Delegation And Bounded Subagents ##########
    ###################################################
    def tool_delegate(self, args):
        if self.depth >= self.max_depth:
            raise ValueError("delegate depth exceeded")
        task = str(args.get("task", "")).strip()
        if not task:
            raise ValueError("task must not be empty")
        child = MiniAgent(
            model_client=self.model_client,
            workspace=self.workspace,
            session_store=self.session_store,
            approval_policy="never",
            max_steps=int(args.get("max_steps", 3)),
            max_new_tokens=self.max_new_tokens,
            depth=self.depth + 1,
            max_depth=self.max_depth,
            read_only=True,
        )
        child.session["memory"]["task"] = task
        child.session["memory"]["notes"] = [clip(self.history_text(), 300)]
        return "delegate_result:\n" + child.ask(task)


def build_welcome(agent, model, host):
    width = max(68, min(shutil.get_terminal_size((80, 20)).columns, 84))
    inner = width - 4
    gap = 3
    left_width = (inner - gap) // 2
    right_width = inner - gap - left_width

    def row(text):
        body = middle(text, width - 4)
        return f"| {body.ljust(width - 4)} |"

    def divider(char="-"):
        return "+" + char * (width - 2) + "+"

    def center(text):
        body = middle(text, inner)
        return f"| {body.center(inner)} |"

    def cell(label, value, size):
        body = middle(f"{label:<9} {value}", size)
        return body.ljust(size)

    def pair(left_label, left_value, right_label, right_value):
        left = cell(left_label, left_value, left_width)
        right = cell(right_label, right_value, right_width)
        return f"| {left}{' ' * gap}{right} |"

    line = divider("=")
    rows = [center(text) for text in WELCOME_ART]
    rows.extend(
        [
            center("MINI CODING AGENT"),
            divider("-"),
            row(""),
            row("WORKSPACE  " + middle(agent.workspace.cwd, inner - 11)),
            pair("MODEL", model, "BRANCH", agent.workspace.branch),
            pair("APPROVAL", agent.approval_policy, "SESSION", agent.session["id"]),
            row(""),
        ]
    )
    return "\n".join([line, *rows, line])


def build_agent(args):
    workspace = WorkspaceContext.build(args.cwd)
    store = SessionStore(Path(workspace.repo_root) / ".mini-coding-agent" / "sessions")
    model = OllamaModelClient(
        model=args.model,
        host=args.host,
        temperature=args.temperature,
        top_p=args.top_p,
        timeout=args.ollama_timeout,
        stream=getattr(args, 'stream', False),
    )
    session_id = args.resume
    if session_id == "latest":
        session_id = store.latest()
    if session_id:
        return MiniAgent.from_session(
            model_client=model,
            workspace=workspace,
            session_store=store,
            session_id=session_id,
            approval_policy=args.approval,
            max_steps=args.max_steps,
            max_new_tokens=args.max_new_tokens,
        )
    return MiniAgent(
        model_client=model,
        workspace=workspace,
        session_store=store,
        approval_policy=args.approval,
        max_steps=args.max_steps,
        max_new_tokens=args.max_new_tokens,
    )


def build_arg_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Minimal coding agent for Ollama models.",
    )
    parser.add_argument("prompt", nargs="*", help="Optional one-shot prompt.")
    parser.add_argument("--cwd", default=".", help="Workspace directory.")
    parser.add_argument("--model", default="qwen3.5:4b", help="Ollama model name.")
    parser.add_argument("--host", default="http://127.0.0.1:11434", help="Ollama server URL.")
    parser.add_argument("--ollama-timeout", type=int, default=300, help="Ollama request timeout in seconds.")
    parser.add_argument("--resume", default=None, help="Session id to resume or 'latest'.")
    parser.add_argument(
        "--approval",
        choices=("ask", "auto", "never"),
        default="ask",
        help="Approval policy for risky tools; auto grants the model arbitrary command execution and file writes.",
    )
    parser.add_argument("--max-steps", type=int, default=6, help="Maximum tool/model iterations per request.")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum model output tokens per step.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature sent to Ollama.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling value sent to Ollama.")
    parser.add_argument("--stream", action="store_true", help="Enable streaming response output.")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    agent = build_agent(args)

    print(build_welcome(agent, model=args.model, host=args.host))

    if args.prompt:
        prompt = " ".join(args.prompt).strip()
        if prompt:
            print()
            try:
                print(agent.ask(prompt, stream_output=args.stream))
            except RuntimeError as exc:
                print(str(exc), file=sys.stderr)
                return 1
        return 0

    while True:
        try:
            user_input = input("\nmini-coding-agent> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("")
            return 0

        if not user_input:
            continue
        if user_input in {"/exit", "/quit"}:
            return 0
        if user_input == "/help":
            print(HELP_DETAILS)
            continue
        if user_input == "/memory":
            print(agent.memory_text())
            continue
        if user_input == "/session":
            print(agent.session_path)
            continue
        if user_input == "/reset":
            agent.reset()
            print("session reset")
            continue

        print()
        try:
            print(agent.ask(user_input, stream_output=args.stream))
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
