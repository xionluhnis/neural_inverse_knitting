import tensorflow as tf
import re
import numpy as np
import tfmpl
import itertools
import pdb
import matplotlib
import io
import textwrap

fn_clipping01 = lambda tensor: tf.fake_quant_with_min_max_args(tensor, min=0., max=1., num_bits=8)
fn_shift_to_min0 = lambda tensor: (tensor - tf.reduce_min(tensor, axis=[1,2], keepdims=True))
fn_normalize_by_max = lambda tensor: tf.divide(tensor, tf.reduce_max(tf.abs(tensor), axis=[1,2,3], keepdims=True) + 1e-10)
fn_normalize_by_max_per_ch = lambda tensor: tf.divide(tensor, tf.reduce_max(tf.abs(tensor), axis=[1,2], keepdims=True) + 1e-10)





def tf_total_var_summary(name_pattern,
                         bmeanstd=True,
                         bminmax=True,
                         bhist=True,
                         black_list_keyword=None):

    if black_list_keyword is None:
        list_var = [
            v for v in tf.trainable_variables()
            if re.search(name_pattern, v.name)
        ]
    else:
        list_var = [
            v for v in tf.trainable_variables()
            if re.search(name_pattern, v.name)
            and not re.search(black_list_keyword, v.name)
        ]

    for var in list_var:
        print(var.name)
        tf_variable_summary(var, bmeanstd, bminmax, bhist, name=name_pattern)

def tf_img_summary(img,
                   bremove_offset=False,
                   bscale_normalize=True,
                   bclip=True,
                   max_outputs=8,
                   name='img'):
    if bremove_offset:
        img = fn_shift_to_min0(img)
    if bscale_normalize:
        img = fn_normalize_by_max(img)
    if not bscale_normalize and bclip:
        img = fn_clipping01(img)

    if img.get_shape()[3] == 2:
        img = tf.concat([
            tf.expand_dims(img[:, :, :, 0], -1),
            tf.expand_dims(img[:, :, :, 1], -1)
        ], 2)
        # img = tf.concat([img, tf.expand_dims(img[:,:,:,1], -1)], 3)
    elif img.get_shape()[3] > 3:
        img = img[:, :, :, :3]

    tf.summary.image(name, img, max_outputs=max_outputs)


def tf_variable_summary(var, name, bmeanstd=True, bminmax=True, bhist=True):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope(name + '/summaries'):
        summaries = []
        if bmeanstd:
            mean = tf.reduce_mean(var)
            s = tf.summary.scalar('mean', mean)
            summaries.append(s)

            s = tf.summary.scalar('meanabs', tf.reduce_mean(tf.abs(var)))
            summaries.append(s)

            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            s = tf.summary.scalar('stddev', stddev)
            summaries.append(s)

        if bminmax:
            s = tf.summary.scalar('max', tf.reduce_max(var))
            summaries.append(s)
            s = tf.summary.scalar('min', tf.reduce_min(var))
            summaries.append(s)

        if bhist:
            s = tf.summary.histogram('histogram', var)
            summaries.append(s)

        return tf.summary.merge(summaries)

@tfmpl.figure_tensor
def tf_keypoint_summary(points):
    # points is BxNx2
    batch_size = points.shape[0]
    figs = tfmpl.create_figures(batch_size, figsize=(4,4))
    try:
        for idx, f in enumerate(figs):
            ax = f.add_subplot(111)
            ax.scatter(points[idx, :, 0], points[idx, :, 1], c='b')
            f.tight_layout()
    except Exception as ex:
        print('-------------')
        print('--- error ---')
        print('-------------')
        print('keypoint summary exception:', ex)
        # pdb.set_trace()

    return figs


# test code
# if i % 10 == 0:  # Record summaries
#   summary, acc = sess.run([merged], feed_dict=feed_dict)
#   test_writer.add_summary(summary, i)



# from https://github.com/tensorflow/tensorboard/issues/227
# def _figure_to_summary(t_img_with_imsz, tag):
#     # summary_image = tf.Summary.Image(height=t_png_encoded[1], 
#     #                                  width=t_png_encoded[2], 
#     #                                  colorspace=4,  # RGB-A
#     #                                  encoded_image_string=t_png_encoded[0])
#     summary_image = tf.Summary.Image(height=t_png_encoded[1], 
#                                      width=t_png_encoded[2], 
#                                      colorspace=4,  # RGB-A
#                                      encoded_image_string=t_png_encoded[0])
#     summary = tf.Summary(value=[tf.Summary.Value(tag=tag, image=summary_image)])
#     return summary

def _plot_confusion_matrix(cm, labels):
    '''
    :param cm: A confusion matrix: A square ```numpy array``` of the same size as self.labels
`   :return:  A ``matplotlib.figure.Figure`` object with a numerical and graphical representation of the cm array
    '''
    FONT_SIZE = 14

    numClasses = len(labels)

    fig = matplotlib.figure.Figure(figsize=(numClasses, numClasses), 
                                    dpi=80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(cm, cmap='Oranges')

    classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in labels]
    classes = ['\n'.join(textwrap.wrap(l, 20)) for l in classes]

    tick_marks = np.arange(len(classes))
    

    ax.set_xlabel('Predicted', fontsize=FONT_SIZE)
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=-90, ha='center', fontsize=FONT_SIZE)
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True Label', fontsize=FONT_SIZE)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, va='center', fontsize=FONT_SIZE)
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()

    for i, j in itertools.product(range(numClasses), range(numClasses)):
        ax.text(j, i, format(cm[i, j], '.1f') if (cm[i, j] != 0) and (cm[i, j] != np.nan) else '-', 
                horizontalalignment="center", 
                verticalalignment='center', 
                color="black", 
                fontsize=FONT_SIZE)
    fig.set_tight_layout(True)

    """
    Converts a matplotlib figure ``fig`` into a TensorFlow Summary object
    that can be directly fed into ``Summary.FileWriter``.
    :param fig: A ``matplotlib.figure.Figure`` object.
    :return: A TensorFlow ``Summary`` protobuf object containing the plot image
                as a image summary.
    """

    # attach a new canvas if not exists
    if fig.canvas is None:
        matplotlib.backends.backend_agg.FigureCanvasAgg(fig)

    def fig2rgb_array(fig):
        fig.canvas.draw()
        buf = fig.canvas.tostring_rgb()
        ncols, nrows = fig.canvas.get_width_height()
        return np.fromstring(buf, dtype=np.uint8).reshape(1, nrows, ncols, 3)

    image = fig2rgb_array(fig)
    # get PNG data from the figure
    # png_buffer = io.BytesIO()
    # fig.canvas.print_png(png_buffer)
    # png_encoded = png_buffer.getvalue()
    # png_buffer.close()

    return image

def tf_summary_confusionmat(t_confusionmat, numlabel, tag='confusion matrix'):
    t_img_confmat = tf.py_func(lambda x: _plot_confusion_matrix(x, 
                                                        [str(x) for x in np.arange(numlabel)]), 
                                [t_confusionmat], 
                                [tf.uint8])
    
    # summary = _figure_to_summary(t_img_confmat, tag='confusion matrix')
    return t_img_confmat[0]


def comp_confusionmat(predictions_batch, labels_batch, num_classes = None, normalized_row = True, name=None):
    with tf.variable_scope('metrics/confusion' + name, tf.AUTO_REUSE):

        # confusion matrix
        confusion_batch = tf.confusion_matrix(labels=tf.reshape(labels_batch, [-1]), 
                                              predictions=tf.reshape(predictions_batch, [-1]),
                                              num_classes=num_classes)
        confusion_batch = tf.cast(confusion_batch, dtype=tf.float32)
        if normalized_row:
            confusion_batch = confusion_batch/(tf.reduce_sum(confusion_batch, axis=1, keepdims=True)+1e-13)

        # if moving_average:

        # calculate moving averages
        # ema = tf.train.ExponentialMovingAverage(decay=0.99)
        # update_op = ema.apply([confusion_batch])
        # confusion_matrix = ema.average(confusion_batch)

        # else:
        # accumulative
        confusion_matrix = tf.Variable(tf.zeros([num_classes, num_classes], dtype=tf.float32),
                                       name='acc_conf_mat', trainable=False)
        confusion_matrix = tf.assign(confusion_matrix, 
                                    0.95*confusion_matrix + 0.05*confusion_batch)
        if normalized_row:
            confusion_matrix = confusion_matrix/(tf.reduce_sum(confusion_matrix, axis=1, keepdims=True)+1e-13)
    return confusion_matrix