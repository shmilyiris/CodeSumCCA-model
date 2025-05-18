import os
import json


def get_project_tree(root_path: str) -> str:
    tree = []
    prefix_chars = {
        "branch": "│   ",
        "tee": "├── ",
        "last": "└── "
    }

    def walk_directory(path: str, prefix: str = "", is_last: bool = True):
        # 获取当前目录下的文件和文件夹（排除隐藏文件）
        items = [item for item in os.listdir(path) if not item.startswith('.')]
        dirs = [item for item in items if os.path.isdir(os.path.join(path, item))]
        files = [item for item in items if os.path.isfile(os.path.join(path, item))]

        # 处理当前目录
        current_dir = os.path.basename(path)
        if path != root_path:  # 根目录不添加前缀
            tree.append(f"{prefix}{prefix_chars['last' if is_last else 'tee']}{current_dir}/")

        # 生成子项
        child_items = dirs + files
        for i, item in enumerate(child_items):
            is_dir = os.path.isdir(os.path.join(path, item))
            is_last_item = (i == len(child_items) - 1)
            new_prefix = prefix + (prefix_chars['branch'] if not is_last else "    ")
            if is_dir:
                walk_directory(
                    os.path.join(path, item),
                    new_prefix,
                    is_last=is_last_item
                )
            else:
                tree.append(f"{new_prefix}{prefix_chars['last' if is_last_item else 'tee']}{item}")

    walk_directory(root_path, is_last=False)  # 初始调用不标记为最后一个节点
    return "\n".join(tree)


def get_prev_summary(model_path: str, file_name: str) -> str:
    summary_file = os.path.join(model_path, f"{file_name}.txt")

    try:
        with open(summary_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return "No previous summary found for this file."
    except Exception as e:
        return f"Error reading summary: {str(e)}"


def get_call_graph(file_name: str) -> str:
    call_graphs_file = "./data/call_graphs.json"

    try:
        with open(call_graphs_file, "r", encoding="utf-8") as f:
            call_graph_data = json.load(f)
            return call_graph_data.get(file_name, {}).get("call_context", "")
    except (FileNotFoundError, json.JSONDecodeError):
        return ""
    except Exception as e:
        return f"Error loading call graph: {str(e)}"


def get_usage_context(file_name: str) -> str:
    usage_file = "./data/usage_contexts.json"

    try:
        with open(usage_file, "r", encoding="utf-8") as f:
            usage_data = json.load(f)
            return usage_data.get(file_name, {}).get("usage_context", "")
    except (FileNotFoundError, json.JSONDecodeError):
        return ""
    except Exception as e:
        return f"Error loading usage context: {str(e)}"


def get_metadata(file_name: str) -> str:
    usage_file = "./data/metadata.json"

    try:
        with open(usage_file, "r", encoding="utf-8") as f:
            usage_data = json.load(f)
            return usage_data.get(file_name, {})
    except (FileNotFoundError, json.JSONDecodeError):
        return ""
    except Exception as e:
        return f"Error loading usage context: {str(e)}"
