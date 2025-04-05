import javalang
from typing import Any, List


class ASTSerializer:
    def serialize_(self, param):
        pass

    def serialize(self, node: Any) -> str:
        if node is None:
            return ""

        # 如果是叶子节点，直接返回值
        if isinstance(node, str):
            return node

        if isinstance(node, javalang.ast.Node):
            # 获取节点类型
            node_type = type(node).__name__
            # 获取节点的属性和值
            children = []
            for attr, child in node.children:
                if isinstance(child, list):
                    # 如果子节点是列表，则递归处理每个子节点
                    children.extend(child)
                else:
                    children.append(child)

            # 将子节点序列化并用括号包裹
            serialized_children = " ".join([self.serialize(child) for child in children if child is not None])
            return f"({node_type} {serialized_children})"

        return ""

    def deserialize(self, serialized_str: str) -> Any:
        tokens = self._tokenize(serialized_str)
        return self._build_ast(tokens)

    def _tokenize(self, serialized_str: str) -> List[str]:
        tokens = []
        current_token = []
        for char in serialized_str:
            if char == '(' or char == ')':
                if current_token:
                    tokens.append("".join(current_token))
                    current_token = []
                tokens.append(char)
            elif char.isspace():
                if current_token:
                    tokens.append("".join(current_token))
                    current_token = []
            else:
                current_token.append(char)

        if current_token:
            tokens.append("".join(current_token))
        return tokens

    def _build_ast(self, tokens: List[str]) -> Any:
        """
        使用递归的方法构建AST。

        Args:
            tokens (List[str]): token列表

        Returns:
            Any: 构建的AST节点
        """
        if not tokens:
            return None

        token = tokens.pop(0)
        if token == "(":
            # 读取节点类型
            node_type = tokens.pop(0)
            children = []
            while tokens[0] != ")":
                child = self._build_ast(tokens)
                if child is not None:
                    children.append(child)
            tokens.pop(0)  # 移除 ")"
            # 返回构建的节点
            return {"type": node_type, "children": children}
        elif token == ")":
            return None
        else:
            return token


# 示例使用
if __name__ == "__main__":
    java_code = """
    public class HelloWorld {
        public static void main(String[] args) {
            System.out.println("Hello, World!");
        }
    }
    """

    # 解析为AST
    ast = javalang.parse.parse(java_code)

    # 序列化
    serializer = ASTSerializer()
    serialized_ast = serializer.serialize(ast)
    print("Serialized AST:", serialized_ast)

    # 反序列化
    deserialized_ast = serializer.deserialize(serialized_ast)
    print("Deserialized AST:", deserialized_ast)
