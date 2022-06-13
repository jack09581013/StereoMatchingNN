import unittest
import torch
import torch.nn.functional as F
import ganet_lib

class CalculatorTestCase(unittest.TestCase):

    def setUp(self):
        self.batch = 4
        self.channels = 4
        self.height = 240 // 2
        self.width = 576 // 2
        self.max_disparity = 192 // 4

    def test_tensor(self):
        CostAggregationTensor = torch.zeros((self.batch, self.channels, 4, self.max_disparity, self.height, self.width)).cuda()
        CostTensor = torch.zeros((self.batch, self.channels, self.max_disparity, self.height, self.width)).cuda()
        SgaWeightTensor = torch.zeros((self.batch, self.channels, 4, 5, self.height, self.width)).cuda()
        MaxTensor = torch.zeros((self.batch, self.channels, 4, self.height, self.width), dtype=torch.uint8).cuda()

        NoChannelCostTensor = torch.zeros((self.batch, self.max_disparity, self.height, self.width)).cuda()
        LgaWeightTensor = torch.zeros((self.batch, 3, 5, 5, self.height, self.width)).cuda()

        ganet_lib.cuda_test(CostAggregationTensor, CostTensor, SgaWeightTensor, MaxTensor,
                       NoChannelCostTensor, LgaWeightTensor)

        self.assertTrue(all(CostAggregationTensor[:, 2, 2, 2, 2, 2] == 2))
        self.assertTrue(all(CostTensor[:, 2, 2, 2, 2] == 2))
        self.assertTrue(all(SgaWeightTensor[:, 2, 2, 2, 2, 2] == 2))
        self.assertTrue(all(MaxTensor[:, 2, 2, 2, 2] == 2))
        self.assertTrue(all(NoChannelCostTensor[:, 2, 2, 2] == 2))
        self.assertTrue(all(LgaWeightTensor[:, 2, 2, 2, 2, 2] == 2))

        CostAggregationTensor[:, 2, 2, 2, 2, 2] = 0
        CostTensor[:, 2, 2, 2, 2] = 0
        SgaWeightTensor[:, 2, 2, 2, 2, 2] = 0
        MaxTensor[:, 2, 2, 2, 2] = 0
        NoChannelCostTensor[:, 2, 2, 2] = 0
        LgaWeightTensor[:, 2, 2, 2, 2, 2] = 0

        self.assertTrue(CostAggregationTensor.view(-1).sum() == 0)
        self.assertTrue(CostTensor.view(-1).sum() == 0)
        self.assertTrue(SgaWeightTensor.view(-1).sum() == 0)
        self.assertTrue(MaxTensor.view(-1).sum() == 0)
        self.assertTrue(NoChannelCostTensor.view(-1).sum() == 0)
        self.assertTrue(LgaWeightTensor.view(-1).sum() == 0)

if __name__ == '__main__':
   unittest.main(verbosity=2)