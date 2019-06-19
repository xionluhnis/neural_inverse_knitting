import numpy as np
import sys, os
import cv2
from PIL import Image

basedir = os.path.dirname(os.path.realpath(__file__))

# New Instructions (17)
INSTRUCTION_CODES = np.arange(17)
instrToIDTable = np.zeros(256, dtype=np.int32) - 1
instrToIDTable[INSTRUCTION_CODES] = np.arange(len(INSTRUCTION_CODES), dtype=np.int32)

def readInstructionsPNG(filename):
    ''' Exact operation over PNG '''
    im = Image.open(filename)
    instructions = np.array(im)

    # Get instruction IDs
    ids = intructionsToIDs(instructions)
    if np.any(ids < 0):
        raise RuntimeError('Invalid instructions!')

    return ids

def intructionsToIDs(instructions):
    ''' Converts actual command codes to our sequential IDs '''
    ids = instrToIDTable[instructions]
    if np.any(ids < 0):
        raise RuntimeError('Invalid instructions!')
    return ids

def idsToInstructions(ids):
    ''' Converts our sequential IDs to actual command codes '''
    if np.any(ids < 0) or np.any(ids >= len(INSTRUCTION_CODES)):
        raise RuntimeError('Invalid IDs!')
    instructions = INSTRUCTION_CODES[ids]
    return instructions

class KnittingViz(object):

    INSTRUCTION_COLORS = np.array([
            [255,   0,  16],
            [43, 206,  72],
            [255, 255, 128],
            [94, 241, 242],
            [0, 129,  69],
            [0,  92,  49],
            [255,   0, 190],
            [194,   0, 136],
            [126,   0, 149],# % 106,   0, 129;
            [96,   0, 112],# %  76,   0,  92;
            [179, 179, 179],
            [128, 128, 128],
            [255, 230,   6],
            [255, 164,   4],
            [0, 164, 255],
            [0, 117, 220],
            [117,  59,  59],
        ], np.float32) / 255

    INSTRUCTION_ICON_FILES = [ 'K.png', 'P.png', 'T.png', 'M.png',
        'FR.png', 'FR.png', 'FL.png', 'FL.png', 
        'BR.png', 'BR.png', 'BL.png', 'BL.png', 
        'XRp.png', 'XRm.png', 'XLp.png', 'XLm.png', 
        'S.png']


    instance = None

    @staticmethod
    def getInstance():
        if KnittingViz.instance is None:
            KnittingViz.instance = KnittingViz()
        return KnittingViz.instance

    def __init__(self):
        self.instructionIcons = [None for x in KnittingViz.INSTRUCTION_ICON_FILES]
        super(KnittingViz, self).__init__()

    def printInstructionsNice(self, ids, tileSize = 20):
        ''' Expects instructions '''
        iconPath = os.path.join(basedir, 'assets', 'instructions')

        rows = []
        for y in range(ids.shape[0]):
            row = []
            for x in range(ids.shape[1]):
                iid = ids[y,x]
                #import pdb; pdb.set_trace()
                tile = np.tile(np.concatenate((KnittingViz.INSTRUCTION_COLORS[iid][::-1], [1.0])), (tileSize, tileSize, 1))
                if self.instructionIcons[iid] is None:
                    icon = cv2.imread(os.path.join(iconPath, KnittingViz.INSTRUCTION_ICON_FILES[iid]), cv2.IMREAD_UNCHANGED).astype(np.float32) / 255
                    col = icon[:,:,0]
                    alpha = icon[:,:,3]
                    mask = np.greater(alpha, 0)
                    sigma = 0.8
                    icon = np.ones((icon.shape[0], icon.shape[1]), np.float32)
                    icon[mask] = (1 - sigma) + sigma * col[mask] * alpha[mask]
                    self.instructionIcons[iid] = icon
                icon = cv2.resize(self.instructionIcons[iid], (tileSize, tileSize), interpolation = cv2.INTER_LINEAR)
                icon = icon.reshape((icon.shape[0], icon.shape[1], 1))
                #import pdb; pdb.set_trace()
                tile *= icon
                tile = tile[:,:,:3]
                row += [tile]
            rows += [np.concatenate(row, axis=1)]
        res = np.concatenate(rows, axis=0)
        res = (res * 255).astype(np.uint8)
        return res

if __name__ == "__main__":

    for i in range(len(sys.argv) - 1):
        fname = sys.argv[i + 1]
        pattern = readInstructionsPNG(fname)
        image = KnittingViz.getInstance().printInstructionsNice(pattern)
        outname='%s_viz.png' % fname
        cv2.imwrite(outname, image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print('Saved %s' % outname)
