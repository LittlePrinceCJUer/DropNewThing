from local_code.base_class.result import result
import pickle

class Result_Loader(result):
    def load(self):
        with open(self.result_destination_folder_path + self.result_destination_file_name,'rb') as f:
            self.data = pickle.load(f)