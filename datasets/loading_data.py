# Copyright 2021 Tencent

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader
from .crowd_dataset import CrowdDataset

# the function to return the dataloader 
def loading_data(args):
    # the augumentations
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225]),
    ])
    # dcreate  the dataset
    test_set = CrowdDataset(root_path=args.data_path, transform=transform)
    test_loader = DataLoader(test_set, batch_size=1, num_workers=4, shuffle=False, drop_last=False)

    return test_loader
