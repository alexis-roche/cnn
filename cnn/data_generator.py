import numpy as np
import pylab as pl

import vii


def set_current_figure(fig):
    return pl.figure(fig.number)


class PatchSelector(object):

    def __init__(self, fimg, size):
        self._size = size
        self._image = vii.load_image(fimg)
        self._x = -1
        self._y = -1
        self._patch = None
        self._label = 0
        self._fig, self._ax = pl.subplots(1)
        self._fig_zoom = pl.figure()

    def run(self, random=False):
        fig = set_current_figure(self._fig)
        self._image.show('pylab')

        if random:
            self._x = np.random.randint(self._image.dims[0] - self._size + 1)
            self._y = np.random.randint(self._image.dims[1] - self._size + 1)

        else:
            self._button = None
            def on_press(event):
                self._button = event.button
                self._x = int(event.xdata - self._size / 2)
                self._y = int(event.ydata - self._size / 2)
            cid = fig.canvas.mpl_connect('button_press_event', on_press)
            wait = True
            while wait:
                pl.waitforbuttonpress()
                wait = False

        self._patch = self._image.bounding_box(self._x, self._x + self._size, self._y, self._y + self._size)
        self._input_label()

    def show_zoom(self):
        # show bounding box
        fig = set_current_figure(self._fig_zoom)
        self._patch.show('pylab')
        pl.plot(self._size / 2, self._size / 2, 'rx', linewidth=20)
        pl.axis((0, self._size - 1, self._size - 1, 0))

    def show(self):
        # show full image with bounding box overlaid
        fig = set_current_figure(self._fig)
        self._image.show('pylab')
        rect = pl.Rectangle((self._x, self._y), self._size, self._size, linewidth=1, edgecolor='r', facecolor='none')
        pl.plot(self._x + self._size / 2, self._y + self._size / 2, 'rx', linewidth=20)
        self._ax.add_patch(rect)
        pl.axis((0, self._image.dims[0], self._image.dims[1], 0))
        pl.show()

    def _input_label(self, show_zoom=True):
        if show_zoom:
            self.show_zoom()
        self.show()
        self._button = None
        def on_press(event):
            self._button = event.button
            return
        fig = set_current_figure(self._fig)
        cid = fig.canvas.mpl_connect('button_press_event', on_press)
        wait = True
        while wait:
            pl.waitforbuttonpress()
            wait = False
        self._label = int(self._button)

    def close(self):
        pl.close(self._fig.number)
        pl.close(self._fig_zoom.number)

    @property
    def data(self):
        return self._patch.get_data()

    @property
    def label(self):
        return self._label
