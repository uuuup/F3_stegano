import cv2
import numpy as np
from PIL import Image
from scipy import fftpack
from itertools import chain
from collections import deque
from dahuffman import HuffmanCodec
from matplotlib import pyplot as plt

DEBUG = True
class ExitLoop(Exception): pass

quant_tbl = np.array( # quantify table
    [[16, 11, 10, 16, 24, 40, 51, 61],
     [12, 12, 14, 19, 26, 58, 60, 55],
     [14, 13, 16, 24, 40, 57, 69, 56],
     [14, 17, 22, 29, 51, 87, 80, 62],
     [18, 22, 37, 56, 68, 109, 103, 77],
     [24, 35, 55, 64, 81, 104, 113, 92],
     [49, 64, 78, 87, 103, 121, 120, 101],
     [72, 92, 95, 98, 112, 100, 103, 99]])

class JPEGCompress(object):
    def __init__(self):
        pass

    def mat2sequence(self, mat):
        """
        convert matrix into one-dimensional array
        :param mat: 8 * 8 matrix
        :return: one-dimensional array
        """

        sequences = [[] for _  in range(15)]
        for i in range(8):
            for j in range(8):
                sequences[i+j].append(mat[i][j])

        res = []
        for i in range(15):
            res += sequences[i] if i % 2 == 1 else reversed(sequences[i])
        return res

    def sequence2mat(self, sequence):
        """
        convert decoded sequence into 8*8 matrix
        :param sequence:decoded sequence, 64
        :return:
        """

        sp = [1, 3, 6, 10, 15, 21, 28, 36, 43, 49, 54, 58, 61, 63]
        sequences = np.split(np.array(sequence), sp)
        for i in range(15):
            if i % 2 == 0:
                sequences[i] = reversed(sequences[i])
        sequence = deque(list(chain(*sequences)))

        mat = np.zeros((8, 8), dtype=int)
        sp = list(range(1, 15))
        for k in range(8):
            for i in range(sp[k]-1, -1, -1):
                mat[sp[k]-i-1][i] = sequence.popleft()
                if k < 7:
                    mat[8+i-sp[k]][7-i] = sequence.pop()
        return mat

    def dct_and_quant(self, img_path):
        """
        DCT transformation and quantify proccess
        :param img_path: path of the original img
        :return: quantified matrix
        """

        img = Image.open(img_path)
        img_arr = np.array(img)
        width, height = img_arr.shape
        img_blocks = [np.hsplit(item, width // 8) for item in np.vsplit(img_arr, height // 8)]
        dct_blocks = [[np.array(fftpack.dctn(block), dtype=int) for block in line] for line in img_blocks]
        quant_blocks = [[np.array(item / quant_tbl, dtype=int) for item in line] for line in dct_blocks]
        return quant_blocks

    def stegano(self, img_path, quant_blocks):
        """
        stegano process
        :param img_path: path of the img to write
        :param quant_blocks:quantified matrix
        :return:written quantified matrix
        """

        # transform target img info bitstream
        img = Image.open(img_path)
        img_arr = np.array(img).flatten()
        bin_seq = ''.join([bin(num)[2:].zfill(8) for num in img_arr])
        bin_seq = deque([int(c) for c in bin_seq])

        for i in range(len(quant_blocks)):
            for j in range(len(quant_blocks[0])):
                block = quant_blocks[i][j]
                width, height = block.shape
                for m in range(height):
                    for n in range(width):
                        if not bin_seq: return quant_blocks
                        if not block[m][n]: continue # original value is 0
                        if abs(block[m][n]) > 1:
                            if abs(block[m][n]) % 2 != bin_seq[0]:
                                if block[m][n] < 0:
                                    block[m][n] += 1
                                else:
                                    block[m][n] -= 1
                            bin_seq.popleft()
                        elif bin_seq[0]: # original value is +/-1
                            bin_seq.popleft()
                        else:
                            block[m][n] = 0

    def reverse_process(self, stegano_blocks):
        """
        reversed process of the quantify and DCT transformation
        :param stegano_blocks:written quantified matrix
        :return:
        """

        iquant_blocks = [[block * quant_tbl for block in line] for line in stegano_blocks]
        idct_blocks = [[np.array(fftpack.idctn(block), dtype=int)//256 for block in line] for line in iquant_blocks]
        stack_arr = np.array(np.vstack([np.hstack(line) for line in idct_blocks]), dtype=np.uint8)
        Image.fromarray(stack_arr).save('reversed_img.bmp')

    def compress_pic(self, quant_blocks):
        """
        compress the matrix data
        :param quant_blocks: written quantified matrix
        :return:huffman tree and encoded data
        """

        sequence = list(chain(*[list(chain(*[self.mat2sequence(mat) for mat in line])) for line in quant_blocks]))
        codec = HuffmanCodec.from_data(sequence)
        encoded = codec.encode(sequence)
        return codec, encoded

    def decompress_pic(self, codec, compress_pic):
        """
        decompress the encode data
        :param codec: huffman tree
        :param compress_pic: encoded data
        :return:decompressed quantified matrix
        """

        decoded = codec.decode(compress_pic)
        arrays = [self.sequence2mat(decoded[i*64: i*64+64]) for i in range(len(decoded)//64)]
        decoded_blocks = [arrays[i*64: i*64+64] for i in range(len(arrays)//64)]
        return decoded_blocks

    def extrct_info(self, decoded_blocks, secret_shape):
        """
        extract hidden info from the decoded quantified matrix
        :param decoded_blocks: decoded quantified matrix
        :param secret_shape: shape of the secret img
        :return:
        """

        bin_seq = []
        try:
            for i in range(len(decoded_blocks)):
                for j in range(len(decoded_blocks[0])):
                    block = decoded_blocks[i][j]
                    width, height = block.shape
                    for m in range(height):
                        for n in range(width):
                            if block[m][n]:
                                bin_seq.append(str(abs(block[m][n])%2))
                                if len(bin_seq) == 8 * secret_shape[0] * secret_shape[1]:
                                    raise ExitLoop
        except ExitLoop:
            bin_seq = np.split(np.array(bin_seq), len(bin_seq)//8)
            img_arr = np.array([int(''.join(item), 2) for item in bin_seq], dtype=np.uint8).reshape(secret_shape)
            Image.fromarray(img_arr).save('extract_img.jpg')

    def stegano_compress(self, basic_path, secret_path):
        """
        run stegano and compress process
        :param basic_path: path of the original img
        :param secret_path: path of the secret img
        :return:
        """

        quant_blocks = self.dct_and_quant(basic_path)
        stegano_blocks = self.stegano(secret_path, quant_blocks)
        codec, encoded = self.compress_pic(stegano_blocks)
        print('size of compressed pic is %.2fkB' % (len(encoded) / 1024))
        return codec, encoded

    def decompress_reverse(self, codec, encoded, secret_shape):
        """
        run decompress and reversed-stegano processs
        :param codec: huffman tree
        :param encoded: encoded data
        :param secret_shape: shape of the secret img
        :return:
        """

        decoded_blocks = self.decompress_pic(codec, encoded)
        self.extrct_info(decoded_blocks, secret_shape)
        self.reverse_process(decoded_blocks)

    def display_result(self, original_path, stegano_path, secret_path, extract_path):
        """
        compare four pictures
        :param original_path: path of original img
        :param stegano_path: path of stegano img
        :param secret_path: path of secret img
        :param extract_path: path of extracted secret img
        :return:
        """

        plt.subplot(2,2,1)
        plt.imshow(cv2.cvtColor(cv2.imread(original_path), cv2.COLOR_BGR2RGB))
        plt.title('original')
        plt.axis('off')

        plt.subplot(2, 2, 2)
        plt.imshow(cv2.cvtColor(cv2.imread(stegano_path), cv2.COLOR_BGR2RGB))
        plt.title('stegano')
        plt.axis('off')

        plt.subplot(2, 2, 3)
        plt.imshow(cv2.cvtColor(cv2.imread(secret_path), cv2.COLOR_BGR2RGB))
        plt.title('secret')
        plt.axis('off')

        plt.subplot(2, 2, 4)
        plt.imshow(cv2.cvtColor(cv2.imread(extract_path), cv2.COLOR_BGR2RGB))
        plt.title('extracted')
        plt.axis('off')

        plt.savefig('compare_result.jpg')
        if DEBUG: plt.show()

if __name__ == '__main__':
    secret_shape = (100, 100)
    Image.open('secret_original.jpg').convert('L').resize(secret_shape, Image.ANTIALIAS).save('secret_grey.jpg')

    handler = JPEGCompress()
    codec, encoded = handler.stegano_compress('basic_grey.bmp', 'secret_grey.jpg')
    handler.decompress_reverse(codec, encoded, secret_shape)
    handler.display_result('basic_grey.bmp', 'reversed_img.bmp', 'secret_grey.jpg', 'extract_img.jpg')
