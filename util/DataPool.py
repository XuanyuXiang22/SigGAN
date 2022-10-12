import random
import torch


class DataPool:
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_data = 0
            self.data = []

    def query(self, datas):
        if self.pool_size == 0:
            return datas

        return_datas = []
        for data in datas:
            data = torch.unsqueeze(data.data, 0)
            if self.num_data < self.pool_size:
                self.num_data = self.num_data + 1
                self.data.append(data)
                return_datas.append(data)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.data[random_id].clone()
                    self.data[random_id] = data
                    return_datas.append(tmp)
                else:
                    return_datas.append(data)
        return_datas = torch.cat(return_datas, 0)

        return return_datas