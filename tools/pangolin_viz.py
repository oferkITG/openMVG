from multiprocessing import Process, Queue

import cv2
import numpy as np

import pangolin
from OpenGL import GL as gl


class DynamicArray(object):
    def __init__(self, shape=3):
        if isinstance(shape, int):
            shape = (shape,)
        assert isinstance(shape, tuple)

        self.data = np.zeros((1000, *shape))
        self.shape = shape
        self.ind = 0

    def clear(self):
        self.ind = 0

    def append(self, x):
        self.extend([x])

    def extend(self, xs):
        if len(xs) == 0:
            return
        assert np.array(xs[0]).shape == self.shape

        if self.ind + len(xs) >= len(self.data):
            self.data.resize(
                (2 * len(self.data), *self.shape), refcheck=False)

        if isinstance(xs, np.ndarray):
            self.data[self.ind:self.ind + len(xs)] = xs
        else:
            for i, x in enumerate(xs):
                self.data[self.ind + i] = x
            self.ind += len(xs)

    def array(self):
        return self.data[:self.ind]

    def __len__(self):
        return self.ind

    def __getitem__(self, i):
        assert i < self.ind
        return self.data[i]

    def __iter__(self):
        for x in self.data[:self.ind]:
            yield x


class Viewer:
    view_thread: Process
    q_pose: Queue
    q_active: Queue
    q_points: Queue
    q_colors: Queue
    q_graph: Queue
    q_camera: Queue
    q_image: Queue
    config = None
    label: str

    def __init__(self, label: str = 'Viewer'):
        self.label = label
        self.q_pose = Queue()
        self.q_active = Queue()
        self.q_points = Queue()
        self.q_colors = Queue()
        self.q_graph = Queue()
        self.q_camera = Queue()
        self.q_image = Queue()
        self.config = None
        self.view_thread = Process(target=self.view)
        self.view_thread.start()

    def drawPlane(self, num_divs=200, div_size=10):

        # Plane parallel to x-z at origin with normal -y
        minx = -num_divs * div_size
        minz = -num_divs * div_size
        maxx = num_divs * div_size
        maxz = num_divs * div_size
        # gl.glLineWidth(2)
        # gl.glColor3f(0.7,0.7,1.0)
        gl.glColor3f(0.7, 0.7, 0.7)
        gl.glBegin(gl.GL_LINES)
        for n in range(2 * num_divs):
            gl.glVertex3f(minx + div_size * n, 0, minz)
            gl.glVertex3f(minx + div_size * n, 0, maxz)
            gl.glVertex3f(minx, 0, minz + div_size * n)
            gl.glVertex3f(maxx, 0, minz + div_size * n)
        gl.glEnd()

    def view(self):
        pangolin.CreateWindowAndBind(self.label, 1024, 768)

        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        panel = pangolin.CreatePanel('menu')
        panel.SetBounds(0.5, 1.0, 0.0, 175 / 1024.)

        # checkbox
        m_follow_camera = pangolin.VarBool('menu.Follow Camera', value=True, toggle=True)
        m_show_points = pangolin.VarBool('menu.Show Points', True, True)
        m_show_keyframes = pangolin.VarBool('menu.Show KeyFrames', True, True)
        m_show_graph = pangolin.VarBool('menu.Show Graph', True, True)
        m_show_image = pangolin.VarBool('menu.Show Image', True, True)

        # button
        m_replay = pangolin.VarBool('menu.Replay', value=False, toggle=False)
        m_refresh = pangolin.VarBool('menu.Refresh', False, False)
        m_reset = pangolin.VarBool('menu.Reset', False, False)

        if self.config is None:
            width, height = 400, 250
            viewpoint_x = 0
            viewpoint_y = -500  # -10
            viewpoint_z = -100  # -0.1
            viewpoint_f = 2000
            camera_width = 1.
        else:
            width = self.config.view_image_width
            height = self.config.view_image_height
            viewpoint_x = self.config.view_viewpoint_x
            viewpoint_y = self.config.view_viewpoint_y
            viewpoint_z = self.config.view_viewpoint_z
            viewpoint_f = self.config.view_viewpoint_f
            camera_width = self.config.view_camera_width

        proj = pangolin.ProjectionMatrix(
            1024, 768, viewpoint_f, viewpoint_f, 512, 389, 0.1, 5000)
        look_view = pangolin.ModelViewLookAt(
            viewpoint_x, viewpoint_y, viewpoint_z, 0, 0, 0, 0, -1, 0)

        # Camera Render Object (for view / scene browsing)
        scam = pangolin.OpenGlRenderState(proj, look_view)

        # Add named OpenGL viewport to window and provide 3D Handler
        dcam = pangolin.CreateDisplay()
        dcam.SetBounds(0.0, 1.0, 175 / 1024., 1.0, -1024 / 768.)
        dcam.SetHandler(pangolin.Handler3D(scam))

        # image
        # width, height = 400, 130
        dimg = pangolin.Display('image')
        dimg.SetBounds(0, height / 768., 0.0, width / 1024., 1024 / 768.)
        dimg.SetLock(pangolin.Lock.LockLeft, pangolin.Lock.LockTop)

        texture = pangolin.GlTexture(width, height, gl.GL_RGB, False, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
        image = np.ones((height, width, 3), 'uint8')

        pose = pangolin.OpenGlMatrix()  # identity matrix
        following = True

        active = []
        replays = []
        graph = []
        loops = []
        mappoints = DynamicArray(shape=(3,))
        colors = DynamicArray(shape=(3,))
        cameras = DynamicArray(shape=(4, 4))

        while not pangolin.ShouldQuit():

            if not self.q_pose.empty():
                pose.m = self.q_pose.get()

            follow = m_follow_camera.Get()
            if follow and following:
                scam.Follow(pose, True)
            elif follow and not following:
                scam.SetModelViewMatrix(look_view)
                scam.Follow(pose, True)
                following = True
            elif not follow and following:
                following = False

            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            gl.glClearColor(1.0, 1.0, 1.0, 1.0)
            dcam.Activate(scam)

            gl.glLineWidth(10)
            gl.glColor3f(0.0, 0.0, 0.0)
            traj_x: np.ndarray = np.array([[.0, .0, .0], [1.0, .0, .0]])
            traj_y: np.ndarray = np.array([[.0, .0, .0], [.0, 1.0, .0]])
            traj_z: np.ndarray = np.array([[.0, .0, .0], [.0, .0, 1.0]])
            gl.glColor3f(1.0, .0, 0.0)
            pangolin.DrawLine(traj_x)  # consecutive
            gl.glColor3f(.0, 1.0, 0.0)
            pangolin.DrawLine(traj_y)  # consecutive
            gl.glColor3f(.0, .0, 1.0)
            pangolin.DrawLine(traj_z)  # consecutive

            # show graph
            if not self.q_graph.empty():
                graph = self.q_graph.get()
                loops = np.array([_[0] for _ in graph if _[1] == 2])
                graph = np.array([_[0] for _ in graph if _[1] < 2])
            if m_show_graph.Get():
                if len(graph) > 0:
                    gl.glLineWidth(1)
                    gl.glColor3f(0.0, 1.0, 0.0)
                    pangolin.DrawLines(graph, 3)
                if len(loops) > 0:
                    gl.glLineWidth(2)
                    gl.glColor3f(0.0, 0.0, 0.0)
                    pangolin.DrawLines(loops, 4)

                gl.glPointSize(4)
                gl.glColor3f(1.0, 0.0, 0.0)
                gl.glBegin(gl.GL_POINTS)
                gl.glVertex3d(pose[0, 3], pose[1, 3], pose[2, 3])
                gl.glEnd()

            # Show mappoints
            if not self.q_points.empty():
                pts, code = self.q_points.get()
                cls, code = self.q_colors.get()
                if code == 1:  # append new points
                    mappoints.extend(pts)
                    colors.extend(cls)
                elif code == 0:  # refresh all points
                    mappoints.clear()
                    mappoints.extend(pts)
                    colors.clear()
                    colors.extend(cls)

            if m_show_points.Get():
                gl.glPointSize(2)
                # easily draw millions of points
                pangolin.DrawPoints(mappoints.array(), colors.array())

                if not self.q_active.empty():
                    active = self.q_active.get()

                gl.glPointSize(3)
                gl.glBegin(gl.GL_POINTS)
                gl.glColor3f(1.0, 0.0, 0.0)
                for point in active:
                    gl.glVertex3f(*point)
                gl.glEnd()

            if len(replays) > 0:
                n = 300
                gl.glPointSize(4)
                gl.glColor3f(1.0, 0.0, 0.0)
                gl.glBegin(gl.GL_POINTS)
                for point in replays[:n]:
                    gl.glVertex3f(*point)
                gl.glEnd()
                replays = replays[n:]

            # show cameras
            if not self.q_camera.empty():
                cams = self.q_camera.get()
                if len(cams) > 20:
                    cameras.clear()
                cameras.extend(cams)

            if m_show_keyframes.Get():
                gl.glLineWidth(1)
                gl.glColor3f(0.0, 0.0, 1.0)
                pangolin.DrawCameras(cameras.array(), camera_width)

            # show image
            if not self.q_image.empty():
                image = self.q_image.get()
                if image.ndim == 3:
                    image = image[::-1, :, ::-1]
                else:
                    image = np.repeat(image[::-1, :, np.newaxis], 3, axis=2)
                image = cv2.resize(image, (width, height))
            if m_show_image.Get():
                texture.Upload(image, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
                dimg.Activate()
                gl.glColor3f(1.0, 1.0, 1.0)
                texture.RenderToViewport()

            if pangolin.Pushed(m_replay):
                replays = mappoints.array()

            if pangolin.Pushed(m_reset):
                m_show_graph.SetVal(True)
                m_show_keyframes.SetVal(True)
                m_show_points.SetVal(True)
                m_show_image.SetVal(True)
                m_follow_camera.SetVal(True)
                follow_camera = True

            if pangolin.Pushed(m_refresh):
                self.q_refresh.put(True)

            pangolin.FinishFrame()
