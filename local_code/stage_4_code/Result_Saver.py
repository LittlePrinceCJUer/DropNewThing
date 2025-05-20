from local_code.base_class.result import result
import pickle

class Result_Saver(result):
    def save(self):
        with open(self.result_destination_folder_path + self.result_destination_file_name,'wb') as f:
            pickle.dump(self.data, f)