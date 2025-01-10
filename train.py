# train gnn

from util.serializer import ASTSerializer
from model.GNN import GCNEncoder

if __name__ == '__main__':
    srlz = ASTSerializer()
    with open("data/sample.txt", 'r+', encoding='UTF-8') as f:
        ast = srlz.deserialize(f.read())
        gnn = GCNEncoder(ast)