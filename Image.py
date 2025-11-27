class Image:
    def __init__(self, value):
        self.value = value
        self.name = None
        self.keypoints = None
        self.descriptors = None
        self.process_image = None

    def set_name(self, name):
        self.name = name

    def set_keypoints(self,keypoints):
        self.keypoints = keypoints

    def set_descriptors(self,descriptors):
        self.descriptors = descriptors

    def set_process_image(self,process_image):
        self.process_image = process_image