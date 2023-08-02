from PIL import Image

import numpy as np
from mmpretrain.apis import ImageClassificationInferencer


# print all system environment variables
import os
from pprint import pprint
pprint(dict(os.environ), width=1)


# ANSI escape code Test
def print_format_table():
    """
    prints table of formatted text format options
    """
    for style in range(8):
        for fg in range(30,38):
            s1 = ''
            for bg in range(40,48):
                format = ';'.join([str(style), str(fg), str(bg)])
                s1 += '\x1b[%sm %s \x1b[0m' % (format, format)
            print(s1)
        print('\n')


print('\x1b[%sm %s \x1b[0m' % ('5;31;46', '\n' + '=' * 29 + ' ANSI TEST ' + '=' * 30))
print_format_table()
print('\x1b[%sm %s \x1b[0m' % ('5;31;46', '=' * 29 + ' ANSI TEST ' + '=' * 30 + '\n'))


model_name = "resnet50_8xb32_in1k"

image_inferencer = ImageClassificationInferencer(model_name)

def post_process(out):
    out['pred_scores'] = out['pred_scores'].tolist()
    return out


class Inferencer():

    def __call__(self, image: Image.Image):
        outputs = image_inferencer(np.asarray(image))
        
        return [ post_process(out) for out in outputs]


inferencer = Inferencer()
