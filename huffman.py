"""
Code for compressing and decompressing using Huffman compression.
"""

from nodes import HuffmanNode, ReadNode


# ====================
# Helper functions for manipulating bytes


def get_bit(byte, bit_num):
    """ Return bit number bit_num from right in byte.

    @param int byte: a given byte
    @param int bit_num: a specific bit number within the byte
    @rtype: int

    >>> get_bit(0b00000101, 2)
    1
    >>> get_bit(0b00000101, 1)
    0
    """
    return (byte & (1 << bit_num)) >> bit_num


def byte_to_bits(byte):
    """ Return the representation of a byte as a string of bits.

    @param int byte: a given byte
    @rtype: str

    >>> byte_to_bits(14)
    '00001110'
    """
    return "".join([str(get_bit(byte, bit_num))
                    for bit_num in range(7, -1, -1)])


def bits_to_byte(bits):
    """ Return int represented by bits, padded on right.

    @param str bits: a string representation of some bits
    @rtype: int

    >>> bits_to_byte("00000101")
    5
    >>> bits_to_byte("101") == 0b10100000
    True
    """
    return sum([int(bits[pos]) << (7 - pos)
                for pos in range(len(bits))])


# ====================
# Functions for compression


def make_freq_dict(text):
    """ Return a dictionary that maps each byte in text to its frequency.

    @param bytes text: a bytes object
    @rtype: dict{int,int}

    >>> d = make_freq_dict(bytes([65, 66, 67, 66]))
    >>> d == {65: 1, 66: 2, 67: 1}
    True
    """

    freq_dict = {}
    for x in text:
        if x in freq_dict:
            freq_dict[x] += 1
        else:
            freq_dict[x] = 1

    return freq_dict


def huffman_tree(freq_dict):
    """ Return the root HuffmanNode of a Huffman tree corresponding
    to frequency dictionary freq_dict.

    @param dict(int,int) freq_dict: a frequency dictionary
    @rtype: HuffmanNode

    >>> freq = {2: 6, 3: 4}
    >>> t = huffman_tree(freq)
    >>> result1 = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> result2 = HuffmanNode(None, HuffmanNode(2), HuffmanNode(3))
    >>> t == result1 or t == result2
    True
    >>> frequence = {3 : 10, 5 : 20, 7 : 30, 11 : 40, 13 : 50}
    """

    def get_value(item):
        """
        Return value from an item.

        @param lst item: constructing_nodes
        @rtype: str

        >>> a = [1, 2, 3]
        >>> get_value(a)
        2
        >>> b = [4, 6, 1]
        >>> get_value(b)
        6
        """
        return item[1]

    # Making leaf nodes
    constructing_nodes = []
    for key, value in freq_dict.items():
        constructing_nodes.append((HuffmanNode(key), value))

    # If there is a single symbol
    if len(constructing_nodes) == 1:
        return HuffmanNode(left=constructing_nodes[0][0])

    # Sorting the leaf nodes by frequency (biggest to smallest)
    constructing_nodes = sorted(constructing_nodes, key=get_value)[::-1]
    while len(constructing_nodes) > 1:
        smallest_freq = constructing_nodes.pop()
        second_smallest_freq = constructing_nodes.pop()

        # Constucting interval node.
        interval_node = HuffmanNode(left=smallest_freq[0],
                                    right=second_smallest_freq[0])
        interval_node_freq = smallest_freq[1] + second_smallest_freq[1]

        # Append the interval node & sort again...
        constructing_nodes.append((interval_node, interval_node_freq))
        constructing_nodes = sorted(constructing_nodes, key=get_value)[::-1]

    return constructing_nodes[0][0]


def get_codes(tree):
    """ Return a dict mapping symbols from tree rooted at HuffmanNode to codes.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: dict(int,str)

    >>> tree = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> d = get_codes(tree)
    >>> d == {3: "0", 2: "1"}
    True
    """

    def get_codes_helper(tree, current_code):
        """
        Return code dictionary with every leaves in a tree

        """

        # Base Case (Checking leaf node)
        code_dict = {}
        if tree.is_leaf():
            code_dict[tree.symbol] = current_code

        # Recursion
        if tree.left is not None:
            code_dict.update(get_codes_helper(tree.left, current_code + "0"))
        if tree.right is not None:
            code_dict.update(get_codes_helper(tree.right, current_code + "1"))

        return code_dict

    # Invalid input check
    if tree is None:
        return None

    # Recursion
    code_dict = {}
    if tree.left is not None:
        code_dict.update(get_codes_helper(tree.left, "0"))
    if tree.right is not None:
        code_dict.update(get_codes_helper(tree.right, "1"))

    return code_dict


def number_nodes(tree):
    """ Number internal nodes in tree according to postorder traversal;
    start numbering at 0.

    @param HuffmanNode tree:  a Huffman tree rooted at node 'tree'
    @rtype: NoneType

    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(None, HuffmanNode(9), HuffmanNode(10))
    >>> tree = HuffmanNode(None, left, right)
    >>> number_nodes(tree)
    >>> tree.left.number
    0
    >>> tree.right.number
    1
    >>> tree.number
    2
    """

    def number_nodes_helper(tree, count):
        """
        Return a number of leaves.

        """

        # Base Case (Leaf node returns without updating the count)
        if tree.is_leaf():
            return count

        # Recursion left -> right (preorder)
        if tree.left is not None:
            count = number_nodes_helper(tree.left, count)
        if tree.right is not None:
            count = number_nodes_helper(tree.right, count)

        # Updating the node number
        tree.number = count
        return count + 1

    # Invalid input check
    if tree is None:
        return None

    # Recursion
    count = 0
    if tree.left is not None:
        count = number_nodes_helper(tree.left, 0)
    if tree.right is not None:
        count = number_nodes_helper(tree.right, count)

    # Updating teh node number
    tree.number = count


def avg_length(tree, freq_dict):
    """ Return the number of bits per symbol required to compress text
    made of the symbols and frequencies in freq_dict, using the Huffman tree.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @param dict(int,int) freq_dict: frequency dictionary
    @rtype: float

    >>> freq = {3: 2, 2: 7, 9: 1}
    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(9)
    >>> tree = HuffmanNode(None, left, right)
    >>> avg_length(tree, freq)
    1.9
    """
    codes = get_codes(tree)

    total_bits = 0
    total_freq = 0
    for key, value in codes.items():
        total_bits += (len(value) * freq_dict[key])
        total_freq += freq_dict[key]

    return total_bits / total_freq


def generate_compressed(text, codes):
    """ Return compressed form of text, using mapping in codes for each symbol.

    @param bytes text: a bytes object
    @param dict(int,str) codes: mappings from symbols to codes
    @rtype: bytes

    >>> d = {0: "0", 1: "10", 2: "11"}
    >>> text = bytes([1, 2, 1, 0])
    >>> result = generate_compressed(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111000']
    >>> text = bytes([1, 2, 1, 0, 2])
    >>> result = generate_compressed(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111001', '10000000']
    """

    result = []
    current_bits = ""
    for x in text:
        code_length = len(codes[x])
        current_bits_length = len(current_bits)
        sum = code_length + current_bits_length
        if sum < 8:
            current_bits += codes[x]
        elif sum == 8:
            current_bits += codes[x]
            result.append(bits_to_byte(current_bits))
            current_bits = ""
        else:
            # First byte segment
            remaining_spot = 8 - current_bits_length
            current_bits += codes[x][:remaining_spot]
            result.append(bits_to_byte(current_bits))

            # Might be more byte segments...
            index = remaining_spot
            while (code_length - index) > 8:
                current_bits = codes[x][index:index+8]
                result.append(bits_to_byte(current_bits))
                index += 8

            # Remainder
            current_bits = codes[x][index:]

    if len(current_bits) > 0:
        result.append(bits_to_byte(current_bits))

    return bytes(result)


def tree_to_bytes(tree):
    """ Return a bytes representation of the tree rooted at tree.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: bytes

    The representation should be based on the postorder traversal of tree
    internal nodes, starting from 0.
    Precondition: tree has its nodes numbered.

    >>> tree = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2]
    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(5)
    >>> tree = HuffmanNode(None, left, right)
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2, 1, 0, 0, 5]
    """

    # Left
    result_bytes = bytes()
    if not tree.left.is_leaf():
        result_bytes += tree_to_bytes(tree.left)

    # Right
    if not tree.right.is_leaf():
        result_bytes += tree_to_bytes(tree.right)

    l_type = 0 if tree.left.is_leaf() else 1
    l_data = tree.left.symbol if tree.left.is_leaf() else tree.left.number
    r_type = 0 if tree.right.is_leaf() else 1
    r_data = tree.right.symbol if tree.right.is_leaf() else tree.right.number
    result_bytes += bytes([l_type, l_data, r_type, r_data])

    return result_bytes


def num_nodes_to_bytes(tree):
    """ Return number of nodes required to represent tree (the root of a
    numbered Huffman tree).

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: bytes
    """
    return bytes([tree.number + 1])


def size_to_bytes(size):
    """ Return the size as a bytes object.

    @param int size: a 32-bit integer that we want to convert to bytes
    @rtype: bytes

    >>> list(size_to_bytes(300))
    [44, 1, 0, 0]
    """
    # little-endian representation of 32-bit (4-byte)
    # int size
    return size.to_bytes(4, "little")


def compress(in_file, out_file):
    """ Compress contents of in_file and store results in out_file.

    @param str in_file: input file whose contents we want to compress
    @param str out_file: output file, where we store our compressed result
    @rtype: NoneType
    """
    with open(in_file, "rb") as f1:
        text = f1.read()
    freq = make_freq_dict(text)
    tree = huffman_tree(freq)
    codes = get_codes(tree)
    number_nodes(tree)

    result = (num_nodes_to_bytes(tree) + tree_to_bytes(tree) +
              size_to_bytes(len(text)))
    result += generate_compressed(text, codes)
    with open(out_file, "wb") as f2:
        f2.write(result)


# ====================
# Functions for decompression


def generate_tree_general(node_lst, root_index):
    """ Return the root of the Huffman tree corresponding
    to node_lst[root_index].

    The function assumes nothing about the order of the nodes in the list.

    @param list[ReadNode] node_lst: a list of ReadNode objects
    @param int root_index: index in the node list
    @rtype: HuffmanNode

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 1, 1, 0)]
    >>> generate_tree_general(lst, 2)
    HuffmanNode(None, HuffmanNode(None, HuffmanNode(10, None, None), \
HuffmanNode(12, None, None)), \
HuffmanNode(None, HuffmanNode(5, None, None), HuffmanNode(7, None, None)))
    """

    root = node_lst[root_index]

    # Constructing left side
    if root.l_type == 1:
        left = generate_tree_general(node_lst, root.l_data)
    else:
        left = HuffmanNode(root.l_data, None, None)

    # Constructing right side
    if root.r_type == 1:
        right = generate_tree_general(node_lst, root.r_data)
    else:
        right = HuffmanNode(root.r_data, None, None)

    return HuffmanNode(None, left, right)


def generate_tree_postorder(node_lst, root_index):
    """ Return the root of the Huffman tree corresponding
    to node_lst[root_index].

    The function assumes that the list represents a tree in postorder.

    @param list[ReadNode] node_lst: a list of ReadNode objects
    @param int root_index: index in the node list
    @rtype: HuffmanNode

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 0, 1, 0)]
    >>> generate_tree_postorder(lst, 2)
    HuffmanNode(None, HuffmanNode(None, HuffmanNode(5, None, None), \
HuffmanNode(7, None, None)), \
HuffmanNode(None, HuffmanNode(10, None, None), HuffmanNode(12, None, None)))
    """

    def generate_tree_helper(node_lst, root_index):
        """ Return a HuffmanNode and index
        """

        root = node_lst[root_index]

        # Constructing right side
        index = root_index
        if root.r_type == 1:
            right, index = generate_tree_helper(node_lst, root_index - 1)
        else:
            right = HuffmanNode(root.r_data, None, None)

        # Constructing left side
        if root.l_type == 1:
            left, index = generate_tree_helper(node_lst, index - 1)
        else:
            left = HuffmanNode(root.l_data, None, None)

        return HuffmanNode(None, left, right), index

    root, _ = generate_tree_helper(node_lst, root_index)
    return root


def generate_uncompressed(tree, text, size):
    """ Use Huffman tree to decompress size bytes from text.

    @param HuffmanNode tree: a HuffmanNode tree rooted at 'tree'
    @param bytes text: text to decompress
    @param int size: how many bytes to decompress from text.
    @rtype: bytes
    """

    def get_text(tree, entire_compressed_text, index, size):
        """ Return text
        """
        # Base case
        if tree.is_leaf():
            return tree.symbol, index
        if index >= size:
            return -1, index

        # Recursion steps...
        if entire_compressed_text[index] == "0":
            return get_text(tree.left, entire_compressed_text, index + 1, size)

        else:
            return get_text(tree.right, entire_compressed_text, index + 1, size)

    # Connecting all the bytes together into a single string.
    index = 0
    entire_compressed_text = ""
    while index < len(text):
        entire_compressed_text += byte_to_bits(text[index])
        index = index + 1

    index = 0
    length = len(entire_compressed_text)
    result_bytes = []
    while index < length and len(result_bytes) < size:
        number, index = get_text(tree, entire_compressed_text, index,
                                 len(entire_compressed_text))
        if number != -1:
            result_bytes.append(number)

    return bytes(result_bytes)


def bytes_to_nodes(buf):
    """ Return a list of ReadNodes corresponding to the bytes in buf.

    @param bytes buf: a bytes object
    @rtype: list[ReadNode]

    >>> bytes_to_nodes(bytes([0, 1, 0, 2]))
    [ReadNode(0, 1, 0, 2)]
    """
    lst = []
    for i in range(0, len(buf), 4):
        l_type = buf[i]
        l_data = buf[i + 1]
        r_type = buf[i + 2]
        r_data = buf[i + 3]
        lst.append(ReadNode(l_type, l_data, r_type, r_data))
    return lst


def bytes_to_size(buf):
    """ Return the size corresponding to the
    given 4-byte little-endian representation.

    @param bytes buf: a bytes object
    @rtype: int

    >>> bytes_to_size(bytes([44, 1, 0, 0]))
    300
    """
    return int.from_bytes(buf, "little")


def uncompress(in_file, out_file):
    """ Uncompress contents of in_file and store results in out_file.

    @param str in_file: input file to uncompress
    @param str out_file: output file that will hold the uncompressed results
    @rtype: NoneType
    """
    with open(in_file, "rb") as f:
        num_nodes = f.read(1)[0]
        buf = f.read(num_nodes * 4)
        node_lst = bytes_to_nodes(buf)
        # use generate_tree_general or generate_tree_postorder here
        tree = generate_tree_general(node_lst, num_nodes - 1)
        size = bytes_to_size(f.read(4))
        with open(out_file, "wb") as g:
            text = f.read()
            g.write(generate_uncompressed(tree, text, size))


# ====================
# Other functions

def improve_tree(tree, freq_dict):
    """ Improve the tree as much as possible, without changing its shape,
    by swapping nodes. The improvements are with respect to freq_dict.

    @param HuffmanNode tree: Huffman tree rooted at 'tree'
    @param dict(int,int) freq_dict: frequency dictionary
    @rtype: NoneType

    >>> left = HuffmanNode(None, HuffmanNode(99), HuffmanNode(100))
    >>> right = HuffmanNode(None, HuffmanNode(101), \
    HuffmanNode(None, HuffmanNode(97), HuffmanNode(98)))
    >>> tree = HuffmanNode(None, left, right)
    >>> freq = {97: 26, 98: 23, 99: 20, 100: 16, 101: 15}
    >>> improve_tree(tree, freq)
    >>> avg_length(tree, freq)
    2.31
    """

    def get_code_length(item):
        """
        Return length of an item
        """
        return len(item[1])

    def get_value(item):
        """
        Return an item with index number 1.
        """
        return item[1]

    def change_symbol(tree, symbol, code, index):
        # Base case
        if len(code) == index:
            tree.symbol = symbol
            return

        # Recursion
        if code[index] is "0":
            change_symbol(tree.left, symbol, code, index + 1)
        else:
            change_symbol(tree.right, symbol, code, index + 1)

    # Sorting the code representation by their length.
    codes = sorted(get_codes(tree).items(), key=get_code_length)

    # Sorting the frequency dictionary from highest to lowest frequency.
    sorted_freq_dict = sorted(freq_dict.items(), key=get_value)[::-1]

    # Give the least length of the code to the
    # symbol having the most frequency.
    for i in range(len(codes)):
        symbol, _ = sorted_freq_dict[i]
        _, code = codes[i]
        change_symbol(tree, symbol, code, 0)




if __name__ == "__main__":
    import python_ta

    python_ta.check_all(config="huffman_pyta.txt")

    import doctest
    doctest.testmod()

    import time

    mode = input("Press c to compress or u to uncompress: ")
    if mode == "c":
        fname = input("File to compress: ")
        start = time.time()
        compress(fname, fname + ".huf")
        print("compressed {} in {} seconds."
              .format(fname, time.time() - start))
    elif mode == "u":
        fname = input("File to uncompress: ")
        start = time.time()
        uncompress(fname, fname + ".orig")
        print("uncompressed {} in {} seconds."
              .format(fname, time.time() - start))
