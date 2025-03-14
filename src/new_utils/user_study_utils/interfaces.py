import wx
import pdb
import numpy as np
import threading
import time
from functools import partial


class VideoDisplay(wx.Panel):
    def __init__(self, parent):
        super(VideoDisplay, self).__init__(parent)

        self.image = None

        self.Bind(wx.EVT_PAINT, self.on_paint)

    def on_paint(self, event):
        dc = wx.BufferedPaintDC(self)
        if self.image is not None:
            dc.DrawBitmap(self.image.ConvertToBitmap(), 50, 50)

    def set_image(self, img):
        self.image = img
        self.Refresh()

    def clear_image(self):
        self.image = None
        self.Refresh()


class VideoPlayerPanel(wx.Panel):
    def __init__(self, parent, human_img_seg):
        super(VideoPlayerPanel, self).__init__(parent)

        self.capture = None
        self.playing = False

        self.video_thread = None

        self.human_img_seg = human_img_seg
        self.size_seg = human_img_seg.shape[0]

        self.init_ui()

    def init_ui(self):
        self.video_display = VideoDisplay(self)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.video_display,
                  proportion=0, 
                  flag=wx.EXPAND | wx.ALL,
                  border=0)

        self.SetSizer(sizer)

        self.video_thread = threading.Thread(target=self.play_video)
        self.video_thread.start()

    def play_video(self):
        id_frame = 0
        pdb.set_trace()
        while True:
            frame = self.human_img_seg[id_frame]
            # wx.Image uses RGB
            h, w = frame.shape[:2]
            img = wx.Image(w, h, frame)
            # Display the image
            wx.CallAfter(self.video_display.set_image, img)
            time.sleep(0.1)  # Adjust the delay based on the video frame rate
            id_frame += 1
            id_frame %= self.size_seg


class AtariMainFrame(wx.Frame):

    def __init__(self, parent, window_title,
                  human_img_seg, agent_act_seg,
                  action_names):
        '''
        human_img_seg.shape: (qbc, size_seg, HWC)
        '''
        wx.Frame.__init__(self, parent, -1, window_title)

        self.human_img_seg = human_img_seg
        self.total_segments = human_img_seg.shape[0]
        self.id_segment = 0
        self.size_segment = human_img_seg.shape[1]
        self.id_frame = 0

        self.frame_h, self.frame_w = self.human_img_seg[-1][0].shape[:2]  # (HWC)
        if self.frame_h > self.frame_w:  # atari
            self.window_h = 800
            self.window_w = 1400
            self.frame_cols = 9
        else:
            self.window_h = 900
            self.window_w = 1700
            self.frame_cols = 3
        self.frame_rows = int(np.ceil((self.size_segment + 2) / (1.0*self.frame_cols)))
        
        # self.scroller = wx.ScrolledWindow(self, -1)
        # self.scroller.SetScrollbars(1, 1, window_w, window_h)

        self.SetBackgroundColour(wx.Colour(224, 224, 224))
        self.SetSize((self.window_w, self.window_h))  # w, h
        self.Center()

        self.agent_act_seg = agent_act_seg
        self.action_names = action_names

        self.CF_list = []
        self.ifall_list = []

        border = 2

        self.sizer_video = wx.BoxSizer(wx.VERTICAL)
        self.video_panel = wx.Panel(self, -1,
                                    # size=(self.window_w//(self.frame_cols+4),
                                    #       self.window_h//(self.frame_rows+2)),
                                    size=(self.frame_w, self.frame_h),
                                    style=wx.SUNKEN_BORDER)
        self.video_panel.SetBackgroundColour(wx.Colour(0, 0, 0))
        # self.image = wx.EmptyImage(self.frame_w, self.frame_h)
        self.video_panel.Bind(wx.EVT_PAINT, self.OnPaintVideo)
        self.label_segment = wx.StaticText(self, -1,
                                            f'Segment {self.id_segment}/{self.total_segments}',
                                            style=wx.ALIGN_CENTER)
        self.sizer_video.Add(self.label_segment,
                             1, wx.EXPAND|wx.ALL, border)
        self.sizer_video.Add(self.video_panel, 1, wx.CENTER|wx.EXPAND|wx.ALL, 0)
        # self.sizer_video.Add(wx.StaticText(self, -1,
        #                                     f'Video',
        #                                     style=wx.ALIGN_CENTER),
        #                                     1, wx.EXPAND|wx.ALL, border)

        rbox_ifall_lblList = ['Yes, give CF(s)', 'No, all good'] 
        self.ifall_panel = wx.Panel(self, -1, style=wx.SUNKEN_BORDER)
        self.rbox_ifall = wx.RadioBox(
                            parent=self.ifall_panel,
                            id=-1,
                            label="Give CF(s) or not",
                            choices=rbox_ifall_lblList,
                            majorDimension=1,
                            style=wx.RA_SPECIFY_COLS,
                        )
        self.rbox_ifall.SetSelection(0)
        
        if self.total_segments == 1:
            btn_next_label = 'Finish'
        else:
            btn_next_label = 'Next'
        self.btn_next_panel = wx.Panel(self, -1, style=wx.SUNKEN_BORDER)
        self.btn_next = wx.Button(parent=self.btn_next_panel,
                                    id=-1,
                                    label=btn_next_label)
        self.btn_next.Bind(wx.EVT_BUTTON, self.OnButtonNext)

        self.sizer_btn = wx.BoxSizer(wx.VERTICAL)
        self.sizer_btn.Add(self.ifall_panel, 1, wx.EXPAND|wx.ALL, border)
        self.sizer_btn.Add(self.btn_next_panel, 1, wx.EXPAND|wx.ALL, border)

        self.sizers_cf_frames = [wx.BoxSizer(wx.VERTICAL) for _ in range(self.size_segment)]
        self.frame_panels = [wx.Panel(self, -1,
                                        # size=(self.window_w//(self.frame_cols+4),
                                        # # size=(-1,
                                        #       self.window_h//(self.frame_rows+2)),
                                        size=(self.frame_w, self.frame_h),
                                        style=wx.SUNKEN_BORDER
                                      ) for _ in range(self.size_segment)]
        self.frame_action_texts = [wx.StaticText(self, -1,
                                                 f'F{_}_{self.action_names[self.agent_act_seg[self.id_segment][_]]}',
                                                 style=wx.ALIGN_CENTER)\
                                    for _ in range(self.size_segment)]
        comb_choices = ["[KEEP]"]
        comb_choices.extend(self.action_names)
        self.cf_comboboxes = [wx.ComboBox(self, -1,
                                         choices=comb_choices,
                                         style=wx.CB_READONLY) for _ in range(self.size_segment)]

        self.sizers_grid = wx.GridSizer(self.frame_rows, self.frame_cols, border, border)
        self.sizers_grid.Add(self.sizer_video, 1, wx.EXPAND|wx.ALL, border)
        self.sizers_grid.Add(self.sizer_btn, 1, wx.EXPAND|wx.ALL, border)
        for id_sizer_cf, size_cf in enumerate(self.sizers_cf_frames):
            self.frame_panels[id_sizer_cf].SetBackgroundColour(wx.Colour(255, 255, 255))
            # t_id_row = (id_sizer_cf + 2) // self.frame_cols
            size_cf.Add(self.frame_action_texts[id_sizer_cf], 1, wx.EXPAND|wx.ALL, border)
            size_cf.Add(self.frame_panels[id_sizer_cf], 1, wx.ALL|wx.EXPAND, 0)
            self.cf_comboboxes[id_sizer_cf].SetSelection(0)
            size_cf.Add(self.cf_comboboxes[id_sizer_cf], 1, wx.EXPAND|wx.ALL, border)
            self.sizers_grid.Add(size_cf, 1, wx.EXPAND|wx.ALL, border)
            self.frame_panels[id_sizer_cf].Bind(wx.EVT_PAINT,
                                                partial(self.OnPaintFrame, id_frame=id_sizer_cf))  # only need to bind on one of these frames
        
        sizer_max = wx.BoxSizer(wx.VERTICAL)
        sizer_max.Add(self.sizers_grid, 1, wx.ALL|wx.EXPAND, border)
        
        self.SetAutoLayout(True)
        self.SetSizer(sizer_max)
        self.Layout()
        
        self.ReDrawVideo()
        self.ReDrawFrames()

        self.Bind(wx.EVT_CLOSE, self.onClose)
    
    def onClose(self, evt):
        if self.id_segment < self.total_segments:
            note_box = wx.MessageDialog(None,
                                        "Haven't check all segments",
                                        "Reminder", wx.OK)
            answer = note_box.ShowModal()
            note_box.Destroy()
            return
        else:
            pass

    def OnPaintVideo(self, evt):
        dc = wx.PaintDC(self.video_panel)  # only used in PaintEvent
        self.PaintVideo(dc)
    
    def ReDrawVideo(self):
        dc = wx.ClientDC(self.video_panel)  # can not be used in PaintEvent
        self.PaintVideo(dc)

    def PaintVideo(self, dc):
        # w, h = self.video_panel.GetSize()
        t_frame = self.human_img_seg[self.id_segment][0].copy()
        # t_image = wx.Image(self.frame_w, self.frame_h)
        # t_image.SetData(np.moveaxis(frame, 0, 1).tobytes())
        # t_image_wxBitmap = t_image.ConvertToBitmap()

        # t_image = cv2.cvtColor(np.uint8(t_frame.copy()), cv2.cv.CV_BGR2RGB)
        # t_image = cv2.cvtColor(np.uint8(t_frame.copy()))
        t_image_wxBitmap = wx.Bitmap.FromBuffer(self.frame_w, self.frame_h, t_frame)
        
        dc.Clear()
        dc.DrawBitmap(t_image_wxBitmap, 0, 0)

    def OnPaintFrame(self, evt, id_frame):
        frame_dc = wx.PaintDC(self.frame_panels[id_frame])
        self.PaintFrame(frame_dc, id_frame)

    def PaintFrame(self, frame_dc, id_frame):
        if self.id_segment >= self.total_segments:
            return
        
        t_frame = self.human_img_seg[self.id_segment][id_frame].copy()
        t_image_wxBitmap = wx.Bitmap.FromBuffer(self.frame_w, self.frame_h, t_frame)

        frame_dc.Clear()
        frame_dc.DrawBitmap(t_image_wxBitmap, 0, 0)

        self.frame_action_texts[id_frame].SetLabel(
            # f"F{id_frame}_{self.action_names[self.agent_act_seg[self.id_segment][id_frame]]}")
            f"S{self.id_segment}F{id_frame}_{self.action_names[self.agent_act_seg[self.id_segment][id_frame]]}")
        
        self.cf_comboboxes[id_frame].SetSelection(0)
    
    def ReDrawFrames(self):
        frame_dcs = [wx.ClientDC(t_frame_panel) for id_frame_panel, t_frame_panel\
                      in enumerate(self.frame_panels)]  # only used in PaintEvent
        self.PaintFrames(frame_dcs)

    def PaintFrames(self, frame_dcs):
        if self.id_segment >= self.total_segments:
            return
        
        # w, h = self.video_panel.GetSize()
        for t_id_frame in range(self.size_segment):
            t_frame = self.human_img_seg[self.id_segment][t_id_frame].copy()
            t_image_wxBitmap = wx.Bitmap.FromBuffer(self.frame_w, self.frame_h, t_frame)

            frame_dcs[t_id_frame].Clear()
            frame_dcs[t_id_frame].DrawBitmap(t_image_wxBitmap, 0, 0)
            
            self.frame_action_texts[t_id_frame].SetLabel(
                # f"F{t_id_frame}_{self.action_names[self.agent_act_seg[self.id_segment][t_id_frame]]}")
                f"S{self.id_segment}F{t_id_frame}_{self.action_names[self.agent_act_seg[self.id_segment][t_id_frame]]}")

            self.cf_comboboxes[t_id_frame].SetSelection(0)
    
    def OnButtonNext(self, evt):
        # if choose Yes give CF, then check there has modified radio choices
        ifall_flag = False if self.rbox_ifall.GetSelection()==0 else True
        self.ifall_list.append(ifall_flag)

        have_cf = False
        cf_this_seg_ls = []
        for t_id_frame in range(self.size_segment):
            cf_action = self.cf_comboboxes[t_id_frame].GetSelection() - 1
            if cf_action != -1:  # select "[KEEP]". NOTE: user in fact can select the agent_actions that they think are correct
                have_cf = True
                cf_this_seg_ls.append((self.id_segment, t_id_frame, cf_action))
        
        if not ifall_flag:
            if not have_cf:
                note_box = wx.MessageDialog(None, "Please give feedbacks on this segment. If you think all agent actions are good, please select 'No' on the top left radio button.",
                                          'Reminder', wx.OK)
                answer = note_box.ShowModal()
                note_box.Destroy()
                return
            else:
                self.CF_list.extend(cf_this_seg_ls)
        else:  # ifall_flag
            if have_cf:
                note_box = wx.MessageDialog(None, "You have given some feedbacks, but you select 'Yes' on the top left radio button.",
                                          'Reminder', wx.OK)
                answer = note_box.ShowModal()
                note_box.Destroy()
                return
            else:
                for t_id_frame in range(self.size_segment):
                    self.CF_list.append((self.id_segment, t_id_frame, self.agent_act_seg[self.id_segment][t_id_frame]))

        self.id_segment += 1
        if self.id_segment == self.total_segments:
            # self.Close()
            self.Destroy()
            return
        elif self.id_segment == self.total_segments - 1:
            self.btn_next.SetLabel("Finish")
        
        self.ReDrawVideo()
        self.ReDrawFrames()

        self.label_segment.SetLabel(f'Segment {self.id_segment}/{self.total_segments}')
        self.rbox_ifall.SetSelection(0)


class AtariUserApp(wx.App):

    def __init__(self,
                 window_title,
                 human_img_seg,
                 agent_act_seg,
                 action_names):
        self.window_title = window_title
        self.human_img_seg = human_img_seg
        self.agent_act_seg = agent_act_seg
        if type(action_names) == dict:
            self.action_names = [action_names[i] for i in range(len(action_names))]
        else:
            self.action_names = action_names
        super().__init__()

    def OnInit(self):
        # self.SetAppName()
        self.Frame = AtariMainFrame(parent=None,
                                    window_title=f"Atari - User Interface: {self.window_title}",
                                    human_img_seg=self.human_img_seg,
                                    agent_act_seg=self.agent_act_seg,
                                    action_names=self.action_names)
        self.Frame.Show()
        return True
    