from collections import ChainMap, defaultdict
from typing import Any
from hashlib import sha1, md5

import matplotlib.pyplot as plt
import networkx as nx


class TreeNode():
    def __init__(
        self,
        parent = None,
        name: str | None = "",
        group: str | None = "",
        value: str | int | float | None = None,
        isDisabled: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._id = md5(hex(id(self)).encode()).hexdigest()
        self._parent = parent
        self.name = name
        self.group = group
        self.value = value
        self.isDisabled = isDisabled
        self.children = []
        
        if parent is None:
            self.index = 0
        else:
            self.attachTo(parent)
            parent.addChild(self)

    def __eq__(self, other):
        if type(other) is type(self):
            return self is other
        elif type(other) is int:
            return self.id == other
        else:
            raise ValueError(f"Not Implemented!!! Comparing {type(self)} and other: {type(other)}")

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        return self.level > other.level

    def __gt__(self, other):
        return self.level < other.level

    def __le__(self, other):
        return not self.__gt__(other)

    def __ge__(self, other):
        return not self.__lt__(other)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        parent = self.parent.id[:4] if self.parent else self.parent
        return f"<Node#{self.id:.4}..,Parent:#{parent}..,Level:{self.level}>"

    def __getitem__(self, index):
        index = int(index)
        if self.isLeaf:
            print("This is Leaf")
            return self
        else:
            return self.children[index]

    def __setitem__(self, index, item):
        index = int(index)
        if self.isLeaf:
            self.children.append(item)
        else:
            self.children[index] = item

    @property
    def id(self):
        return self._id

    @property
    def parent(self):
        return self._parent
    @parent.setter
    def parent(self, value):
        self._parent = value

    @property
    def isRoot(self):
        return True if self.parent is None else False

    @property
    def isLeaf(self):
        return True if len(self.children) == 0 else False

    @property
    def level(self):
        parent = self.parent
        level = 0
        while parent is not None:
            level += 1
            parent = parent.parent

        return level

    def isGroup(self, group: str):
        return self.groop == group

    def isLevel(self, lv: int):
        return self.level == lv

    def isChild(self, other):
        if type(other) is type(self):
            return self in other.children
        else:
            raise ValueError(f"Not Implemented!!! Comparing {type(self)} and other: {type(other)}")

    def isParent(self, other):
        if type(other) is type(self):
            return self is other.parent
        else:
            raise ValueError(f"Not Implemented!!! Comparing {type(self)} and other: {type(other)}")

    def isSibling(self, other):
        if type(other) is type(self):
            return self.parent is other.parent
        else:
            raise ValueError(f"Not Implemented!!! Comparing {type(self)} and other: {type(other)}")

    def addChild(self, other):
        self.children.append(other)
        other.parent = self

        return self

    def attachTo(self, other):
        other.children.append(self)
        self.parent = other

        return self

    def cutOff(self):
        self.parent.children.remove(self)
        self.parent = None

        return self

    def reduce(self):
        self.parent.children.remove(self)
        for child in self.children:
            self.parent.children.append(child)
            child.parent = self.parent

        self.parent = None
        self.children = []
        
        return self

    def view(self, depth: int = 0):
        pass
        

class Tree():
    def __init__(self, node: TreeNode | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.node = node
        self.G = nx.DiGraph()
        self.d = {node.id: node}

        if node:
            self.G, self.d = self._makeTree(node, self.G, self.d)

    def _makeTree(self, node: TreeNode, G, d):
        if node.isLeaf:
            return G, d
        else:
            for c in node.children:
                G.add_node(c.id)
                G.add_edge(node.id, c.id)
                d[c.id] = c
                G, d = self._makeTree(c, G, d)
            return (G, d)

    def show(self, node: TreeNode|None = None, label: str = ""):
        if node:
            G, d = self._makeTree(node, nx.DiGraph(), {node.id: node})
        else:
            G, d = self.G, self.d
        
        pos = nx.drawing.nx_agraph.graphviz_layout(G, prog="dot")

        #fig = plt.figure(figsize=(10, 10), dpi=300)
        #ax = fig.add_subplot(1, 1, 1)
        
        nx.draw(
            G,
            #ax = ax,
            pos = pos,
            with_labels = False,
            arrows = False,
            node_size = 100,
            node_shape = 'o',
            width = 0.5,
        )

        if label == "value":
            labels = { key: value.value for key, value in d.items() }
        else:
            labels = { key: "" for key, valye in d.items() }

        nx.draw_networkx_labels(
            G,
            pos = pos,
            labels = labels,
            font_size = 11,
            font_color = "orange",
        )
            


           


