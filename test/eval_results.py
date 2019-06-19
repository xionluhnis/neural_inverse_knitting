import numpy as np
import sys, os, re, time, json, glob, math, argparse, shutil, csv
import pickle
import cv2
import scipy.io as sio
import scipy.interpolate as sci
import scipy.ndimage.filters as scf
import scipy.ndimage
from PIL import Image

import skimage.measure as skim

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

basedir = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser(description='Experiment Evaluation')
parser.add_argument('method', help='Method name')
parser.add_argument('--nice', type=str2bool, nargs='?', const=True, default=False, help="Sink nice.")
args = parser.parse_args()

def printWrite(fid, txt):
    fid.write(txt)
    print(txt)

class ExperimentEval(object):

    ROOT_PATH = os.path.join(basedir, 'test', 'results')
    
    # New Instructions (17)
    INSTRUCTION_CODES = np.arange(17)


    def __init__(self):
        self.instrToIDTable = np.zeros(256, dtype=np.int32) - 1
        self.instrToIDTable[ExperimentEval.INSTRUCTION_CODES] = np.arange(len(ExperimentEval.INSTRUCTION_CODES), dtype=np.int32)

        self.syntaxT = []
        for i in range(4):
            matname = os.path.join(basedir, '..', 'dataset', 'syntax', 'T' + str(i+1) + '.txt')
            self.syntaxT += [np.loadtxt(matname, delimiter = ',')]

        super(ExperimentEval, self).__init__()


    def evalAll(self):
        dirs = os.listdir(os.path.join(ExperimentEval.ROOT_PATH, 'programs'))
        dirs = np.array(dirs, np.object)
        dirs.sort()

        methods = []
        results = {}
        inFig = []

        for i,method in enumerate(dirs):
            print('[%d/%d] %s' % (i, len(dirs), method))
            dirPath = os.path.join(ExperimentEval.ROOT_PATH, 'programs', method)
            if not os.path.isdir(dirPath) or method == 'gt':
                continue
            
            res = self.eval(method, silent = True)
            if res['acc'] < 0:
                print('====>\n[ExperimentEval] Invalid data for %s => SKIP\n<=====' % method)
                continue

            results[method] = res
            methods += [method]

            inFig += [not (method in ['knitting_cycle_real3', 'knitting_pix2pix_real3', 'seg_real3'])]

            if len(methods) == 3:
                #break
                pass

        methods = np.array(methods, np.object)

        metrics = ['Accuracy', 'Acc. (fg)', 'Accuracy MIL', 'Acc. (fg) MIL', 'Syntax Coherency', 'SSIM', 'PSNR [dB]', 'SSIM MIL', 'PSNR MIL [dB]']
        mKeys = ['acc', 'acc_fg', 'acc_mil', 'acc_fg_mil', 'syntax', 'ssim', 'psnr', 'ssim_mil', 'psnr_mil']

        with open(os.path.join(ExperimentEval.ROOT_PATH, 'summary.txt'), 'w') as fid:
            printWrite(fid, '--------------\n----------------\nOverall results:')
            printWrite(fid, '% 28s\t|\t% 10s\t|\t% 10s\t|\t% 10s\t|\t% 10s\t|\t% 10s\t|\t% 10s\t|\t% 10s|\t% 10s\t|\t% 10s' % tuple(['Method'] + metrics))
            printWrite(fid, '--------------------------------------------------------------------------------------------------------------------------')
            for i,method in enumerate(methods):
                res = results[method]
                vals = [res[k] for k in mKeys]
                printWrite(fid, '% 28s\t|\t% 8.2f%%\t|\t% 8.2f%%\t|\t% 8.2f%%\t|\t% 8.2f%%\t|\t% 8.3f\t|\t% 8.3f\t|\t% 9.2f|\t% 8.3f\t|\t% 9.2f' % tuple([method] + vals))

        for i,m in enumerate(metrics):
            
            vals = []
            for method in methods:
                vals += [results[method][mKeys[i]]]
            vals = np.array(vals)[inFig]
            x = np.arange(len(vals))

            ymin = max(np.min(vals) - (np.max(vals) - np.min(vals)) * 1.0, 0.0)
            ymax = np.max(vals)

            #fig = plt.figure(figsize=(16,7))
            fig, ax = plt.subplots(figsize=(16,7))
            plt.title(m)
            plt.bar(x, vals)
            plt.xticks(x, methods[inFig], rotation=90)
            plt.ylim(ymin, ymax)
            #fig.subplots_adjust(bottom=0.4) 
            fig.tight_layout()
            
            plt.savefig(os.path.join(ExperimentEval.ROOT_PATH, '%s.png' % mKeys[i]), dpi=fig.dpi, transparent=False)
            #plt.show()
            
            #import pdb; pdb.set_trace()

        # Write CSV
        with open(os.path.join(ExperimentEval.ROOT_PATH, 'summary.csv'), 'w', newline='') as fid:
            writer = csv.writer(fid, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['Method'] + metrics)
            for i,method in enumerate(methods):
                res = results[method]
                vals = [res[k] for k in mKeys]
                writer.writerow([method] + vals)
        with open(os.path.join(ExperimentEval.ROOT_PATH, 'summary_per_instr.csv'), 'w', newline='') as fid:
            writer = csv.writer(fid, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            instr_names = list(map(lambda n: str(n), range(17)))
            writer.writerow(['Method'] + instr_names + instr_names)
            for i,method in enumerate(methods):
                res = results[method]
                writer.writerow([method] + res['per_instr']['acc'] + res['per_instr']['acc_mil'])

        # Write matlab
        meta = results
        for res in meta.values():
            del res['per_instr']
        meta['method'] = methods
        sio.savemat(os.path.join(ExperimentEval.ROOT_PATH, 'summary.mat'), meta)
        print('DONE')




    def eval(self, method, silent = False):
        rootPath = ExperimentEval.ROOT_PATH
        predPath = os.path.join(rootPath, 'programs', method)
        renderPath = os.path.join(rootPath, 'renderings', method)
        if args.nice:   
            nicePath = preparePath(os.path.join(rootPath, 'nice', method))

        print('[ExperimentEval] Evaluating %s...' % predPath)

      
        gtPath = os.path.join(rootPath, 'programs', 'gt')
        gtRenderPath = os.path.join(rootPath, 'renderings', 'gt')

        items = glob.glob(os.path.join(predPath, '*.png'))
        items = np.array(items, np.object)
        items = np.array([re.match('.*\/([^\/]+)\.png$', x).group(1) for x in items], np.object)
        items.sort()

        acc = RatioMeter()
        accFg = RatioMeter()
        accMIL = RatioMeter()
        accFgMIL = RatioMeter()
        perInstr = []
        perInstrMIL = []
        for ins in range(17):
            perInstr += [ RatioMeter() ]
            perInstrMIL += [ RatioMeter() ]
        syntax = []
        resSSIM = []
        resPSNR = []
        resSSIMMIL = []
        resPSNRMIL = []

        t0 = 0
        for i,item in enumerate(items):
            if time.time() - t0 >= 1:
                #print('[ExperimentEval] [%s/%s] Evaluating %s [%d/%d] (%.2f%%)...' % (dataType, split, item, i, len(items), i / len(items) * 100))
                t0 = time.time()

            # Read files
            gtFile = os.path.join(gtPath, '%s.png' % item)
            predFile = os.path.join(predPath, '%s.png' % item)

            gtIds = self.readInstructionsPNG(gtFile)
            predIds = self.readInstructionsPNG(predFile)
            
            # Eval INSTRUCTIONS
            correct, nPixels, correctNZ, nPixelsNZ, correctIN, nPixelsIN = self.computeError(gtIds, predIds, useMIL = False)
            acc.add(correct, nPixels)
            accFg.add(correctNZ, nPixelsNZ)
            for ins in range(17):
                perInstr[ins].add(correctIN[ins], nPixelsIN[ins])

            correct, nPixels, correctNZ, nPixelsNZ, correctIN, nPixelsIN = self.computeError(gtIds, predIds, useMIL = True)
            accMIL.add(correct, nPixels)
            accFgMIL.add(correctNZ, nPixelsNZ)
            for ins in range(17):
                perInstrMIL[ins].add(correctIN[ins], nPixelsIN[ins])

            syntaxErrs, _ = self.syntaxError(predIds)
            syntax += [ syntaxErrs ]


            msg = '[% 3d/% 3d]\t% 20s' % (i, len(items), item)
            msg += '\t% 3.2f%% (%.2f%%)\t% 3.2f%% (%.2f%%)' % (acc.last() * 100, acc.mean() * 100, accFg.last() * 100, accFg.mean() * 100)
            msg += '\t% 3.2f%% (%.2f%%)\t% 3.2f%% (%.2f%%)' % (accMIL.last() * 100, accMIL.mean() * 100, accFgMIL.last() * 100, accFgMIL.mean() * 100)
            msg += ' |\t% .3f (%.3f)' % (syntax[-1], np.nanmean(syntax))

            # Eval RENDER
            gtRenderFile = os.path.join(gtRenderPath, '%s.png' % item)
            predRenderFile = os.path.join(renderPath, '%s.png' % item)
            if os.path.isfile(predRenderFile):
                gtRender = cv2.imread(gtRenderFile)[...,0].astype(np.float32) / 255
                predRender = cv2.imread(predRenderFile)[...,0].astype(np.float32) / 255

                #rmse = np.linalg.norm(gtRender - predRender)
                ssim = self.imageMetricMIL(gtRender, predRender, skim.compare_ssim, useMIL = False)
                psnr = self.imageMetricMIL(gtRender, predRender, skim.compare_psnr, useMIL = False)
                ssimMIL = self.imageMetricMIL(gtRender, predRender, skim.compare_ssim, useMIL = True)
                psnrMIL = self.imageMetricMIL(gtRender, predRender, skim.compare_psnr, useMIL = True)

                resSSIM += [ssim]
                resPSNR += [psnr]
                resSSIMMIL += [ssimMIL]
                resPSNRMIL += [psnrMIL]

                msg += ' |\tSSIM = %.3f (%.3f) |\tPSNR = %.2f (%.2f) dB' % (
                    ssim, np.mean(resSSIM),
                    psnr, np.mean(resPSNR),
                )
                msg += ' |\tSSIM_mil = %.3f (%.3f) |\tPSNR_mil = %.2f (%.2f) dB' % (
                    ssim, np.mean(resSSIMMIL),
                    psnr, np.mean(resPSNRMIL),
                )

            if not silent:
                print(msg)

            if args.nice:       
                predNice = KnittingViz.getInstance().printInstructionsNice(predIds)
                cv2.imwrite(os.path.join(nicePath, '%s.png' % item), predNice, [cv2.IMWRITE_JPEG_QUALITY, 95]) # nice viz

        res = {
            'acc': (acc.mean() * 100),
            'acc_fg': (accFg.mean() * 100),
            'acc_mil': (accMIL.mean() * 100),
            'acc_fg_mil': (accFgMIL.mean() * 100),
            'syntax': np.nanmean(syntax),
            'ssim': (np.mean(resSSIM) if len(resSSIM) > 0 else -1),
            'psnr': (np.mean(resPSNR) if len(resPSNR) > 0 else -1),
            'ssim_mil': (np.mean(resSSIMMIL) if len(resSSIMMIL) > 0 else -1),
            'psnr_mil': (np.mean(resPSNRMIL) if len(resPSNRMIL) > 0 else -1),
            'per_instr': {
                'acc':      list(map(lambda x: (x.mean() * 100), perInstr)),
                'acc_mil':  list(map(lambda x: (x.mean() * 100), perInstrMIL)),
            },
        }

        print('-------------------------------')
        print('Overall accuracy: %.2f%%' % res['acc'])
        print('Foreground accuracy: %.2f%%' % res['acc_fg'])
        print('Overall accuracy (MIL): %.2f%%' % res['acc_mil'])
        print('Foreground accuracy (MIL): %.2f%%' % res['acc_fg_mil'])
        print('Syntax: %.3f' % res['syntax'])
        print('Render SSIM: %.2f' % res['ssim'])
        print('Render PSNR: %.3f dB' % res['psnr'])
        print('Render SSIM (MIL): %.2f' % res['ssim_mil'])
        print('Render PSNR (MIL): %.3f dB' % res['psnr_mil'])

        return res



    def computeError(self, gtIds, predIds, useMIL = False):
        idCounts = np.bincount(gtIds.flatten())
        topGT = np.argmax(idCounts)
        
        bestScore = 0
        bestShift = [0, 0]
        if useMIL:
            predIds = predIds[1:-1,1:-1]
            for y in range(-1, 2):
                for x in range(-1, 2):
                    shift = [y, x]
                    gtIdsShifted = scipy.ndimage.shift(gtIds, shift, order=0, mode='nearest')
                    gtIdsShifted = gtIdsShifted[1:-1,1:-1]
                    correct = np.sum(np.equal(gtIdsShifted, predIds))
                    if correct > bestScore:
                        bestScore = correct
                        bestShift = shift
        else:
            bestShift = [0, 0]

        #print(bestShift)
        gtIdsShifted = scipy.ndimage.shift(gtIds, bestShift, order=0, mode='nearest')
        if useMIL:
            gtIdsShifted = gtIdsShifted[1:-1,1:-1]
            pass

        maskNonZero = np.not_equal(gtIdsShifted, topGT)

        correct = np.sum(np.equal(gtIdsShifted, predIds))
        nPixels = predIds.shape[0] * predIds.shape[1]

        correctNZ = np.sum(np.logical_and(np.equal(gtIdsShifted, predIds), maskNonZero))
        nPixelsNZ = np.sum(maskNonZero)

        correctIN = np.zeros(17)
        nPixelsIN = np.zeros(17)
        for ins in range(17):
            maskIns = gtIdsShifted == ins
            correctIN[ins] = np.sum(np.logical_and(np.equal(gtIdsShifted, predIds), maskIns))
            nPixelsIN[ins] = np.sum(maskIns)
        
        return correct, nPixels, correctNZ, nPixelsNZ, correctIN, nPixelsIN

    def imageMetricMIL(self, gt, pred, metric, useMIL = False):
        shiftSize = gt.shape[0] // 20
        bestScore = 0
        bestShift = [0, 0]
        if useMIL:
            pred = pred[shiftSize:-shiftSize,shiftSize:-shiftSize] # crop
            for y in range(-1, 2):
                for x in range(-1, 2):
                    shift = np.array([y, x])
                    gtShifted = scipy.ndimage.shift(gt, shift * shiftSize, order=0, mode='nearest')
                    gtShifted = gtShifted[shiftSize:-shiftSize,shiftSize:-shiftSize]
                    score = metric(gtShifted, pred)
                    if score > bestScore:
                        bestScore = score
                        bestShift = shift
            gtShifted = scipy.ndimage.shift(gt, bestShift * shiftSize, order=0, mode='nearest')
            gtShifted = gtShifted[shiftSize:-shiftSize,shiftSize:-shiftSize]
        else:
            gtShifted = gt

        return metric(gtShifted, pred)


        
    def syntaxError(self, instr, diffonly = True):
        def get_one_hot(img):
            classes = []
            for i in range(17):
                classes.append(img == i)
            return np.stack(classes, axis = 2)

        dx = [ -1, 0, 1, -1, 1, -1, 0, 1]
        dy = [ -1, -1, -1, 0, 0, 1, 1, 1]
        src_from = { -1: 1, 0: 0, 1: 0 }
        trg_from = { -1: 0, 0: 0, 1: 1 }
        rng_size = { -1: 19, 0: 20, 1: 19 }

        total = 0
        invalid = 0
        for i in range(4): # only go over the first half (as the other has the same errors)
            T = self.syntaxT[i]
            # select target slice of instructions
            ys = src_from[dy[i]]
            yt = trg_from[dy[i]]
            yn = rng_size[dy[i]]
            xs = src_from[dx[i]]
            xt = trg_from[dx[i]]
            xn = rng_size[dx[i]]
            instr_src = instr[ys:ys+yn, xs:xs+xn]
            instr_trg = instr[yt:yt+yn, xt:xt+xn]

            # accuracy
            if diffonly:
                mask = (instr_src != instr_trg).astype(np.int32)
            else:
                mask = (instr_src >= 0).astype(np.int32)
            total += np.sum(mask)
            P = 1 - (T >= 1)
            instr_src = get_one_hot(instr_src)
            instr_trg = get_one_hot(instr_trg)
            inval_msk = np.einsum('hwi,ij,hwj->hw', instr_src, P, instr_trg)
            inval_msk *= mask
            invalid += np.sum(inval_msk)
            # print(np.array2string(inval_msk[:, 3:-2]))

        # print(np.array2string(instr[:, 3:-2]))
        # print('%d / %d' % (invalid, total))
        #import pdb; pdb.set_trace()
        return invalid, 1.0 - invalid / total

        
    def readInstructionsPNG(self, filename):
        ''' Exact operation over PNG '''
        im = Image.open(filename)
        instructions = np.array(im)

        # Get instruction IDs
        ids = self.intructionsToIDs(instructions)
        if np.any(ids < 0):
            raise RuntimeError('Invalid instructions!')

        return ids

    def intructionsToIDs(self, instructions):
        ''' Converts actual command codes to our sequential IDs '''
        ids = self.instrToIDTable[instructions]
        if np.any(ids < 0):
            raise RuntimeError('Invalid instructions!')
        return ids

    def idsToInstructions(self, ids):
        ''' Converts our sequential IDs to actual command codes '''
        if np.any(ids < 0) or np.any(ids >= len(ExperimentEval.INSTRUCTION_CODES)):
            raise RuntimeError('Invalid IDs!')
        instructions = ExperimentEval.INSTRUCTION_CODES[ids]
        return instructions


def preparePath(path, clear = False):
    if not os.path.isdir(path):
        os.makedirs(path, 0o777)
    if clear:
        files = os.listdir(path)
        for f in files:
            fPath = os.path.join(path, f)
            if os.path.isdir(fPath):
                shutil.rmtree(fPath)
            else:
                os.remove(fPath)

    return path

class RatioMeter(object):
    def __init__(self):
        self.sum = 0
        self.count = 0
        super(RatioMeter, self).__init__()

    def add(self, val, n):
        self.lastVal = val
        self.lastN = n

        self.sum += val
        self.count += n

    def last(self):
        return self.lastVal / self.lastN if self.lastN > 0 else -1

    def mean(self):
        return self.sum / self.count if self.count > 0 else -1



class KnittingViz(object):

    # INSTRUCTION_COLORS = {
    #     0: [0, 0, 0],
    #     1: [255, 0, 0],
    #     2: [0, 255, 0],
    #     3: [255, 255, 0],
    #     4: [0, 0, 255],
    #     5: [255, 0, 255],
    #     6: [0, 255, 255],
    #     7: [255, 255, 255],
    #     8: [74, 137, 153],
    #     9: [108, 36, 144],
    #     10: [180, 180, 216],
    #     11: [255, 103, 189],
    #     12: [144, 108, 180],
    #     13: [153, 153, 153],
    #     14: [207, 144, 192],
    #     15: [128, 128, 255],
    #     16: [81, 255, 222],
    #     40: [127,0,127],
    #     50: [220,118,117],
    #     61: [0,255,255],
    #     62: [50,233,233],
    #     63: [50,202,233],
    #     64: [53,175,237],
    #     71: [255,255,255],
    #     72: [0,160,160],
    #     73: [183,188,188],
    #     74: [197,174,183],
    #     81: [74,137,153],
    #     82: [109,165,180],
    #     83: [14,192,207],
    #     84: [0,102,255],
    # }
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
        iconPath = os.path.join(basedir, 'test', 'assets', 'instructions')

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
    method = args.method
    ex = ExperimentEval()

    if method == 'all':
        ex.evalAll()
    else:
        ex.eval(method)




    
