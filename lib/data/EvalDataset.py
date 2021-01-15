from .TrainDataset import TrainDataset

class EvalDataset(TrainDataset):
    @staticmethod
    def modify_commandline_options(parser):
        return parser

    def __init__(self, opt):
        super(EvalDataset, self).__init__(opt, phase='eval')
        self.opt = opt

    def __len__(self):
        return 1

    def setBoundingBox(self, min, max):
        self.B_MIN = min
        self.B_MAX = max

    def set_views(self, views):
        self.views = views

    def get_item(self, index):
        subject = 'real_blueprint_test'

        res = {
            'name': subject,
            'b_min': self.B_MIN,
            'b_max': self.B_MAX,
            'samples': None,
            'labels': None,
            'samples_normals': None,
            'normals': None
        }

        render_data = self.get_render(subject, num_views=self.num_views)
        res.update(render_data)
        return res

    def __getitem__(self, index):
        return self.get_item(index)
