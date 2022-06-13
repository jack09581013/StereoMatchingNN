import profile
import CSPN.cspn as cspn
import GANet.GANet_small_deep as ganet_small_deep
import GANet.GANet_small as ganet_small
import GANet.GANet_deep as ganet_deep

class GANet_small_deep_fine_tune(profile.Profile):
    def get_model(self, max_disparity):
        return ganet_small_deep.GANet_small_deep(max_disparity)

    def version_file_path(self):
        return f'../../model/save/GANet_small_deep_fine_tune'

    def __str__(self):
        return 'GANet_small_deep'