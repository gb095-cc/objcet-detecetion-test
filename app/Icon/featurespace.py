import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 加载 CIFAR-10 数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True)

# 使用预训练的 ResNet-18 模型
net = models.resnet18(weights=False, num_classes=10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 训练模型
for epoch in range(2):  # 训练两个epoch
    for inputs, labels in trainloader:
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# 提取中间层特征
def extract_features(model, data):
    features = []
    with torch.no_grad():
        for inputs, _ in data:
            x = model.conv1(inputs)
            x = model.bn1(x)
            x = model.relu(x)
            x = model.maxpool(x)
            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3(x)
            x = model.layer4(x)
            x = torch.flatten(x, 1)
            features.append(x)
    return torch.cat(features)

# 原始特征空间可视化
features = extract_features(net, trainloader)
features_np = features.numpy()

# 使用 PCA 进行特征降维
pca = PCA(n_components=2)
features_pca = pca.fit_transform(features_np)

# 原始特征空间图
plt.figure(figsize=(8, 6))
plt.scatter(features_pca[:, 0], features_pca[:, 1], s=10, cmap='viridis')
plt.title('Feature Space before Perturbation')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid()
plt.savefig('feature_space_before.png')
plt.close()

# 添加对抗扰动 (FGSM 方法)
def fgsm_attack(image, epsilon, gradient):
    sign_gradient = gradient.sign()
    perturbed_image = image + epsilon * sign_gradient
    return perturbed_image

# 对图像进行扰动，并提取扰动后的特征
perturbed_features = []
for inputs, labels in trainloader:
    inputs.requires_grad = True
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    net.zero_grad()
    loss.backward()
    inputs_grad = inputs.grad.data
    perturbed_data = fgsm_attack(inputs, epsilon=0.1, gradient=inputs_grad)
    
    # 提取扰动后图像的特征
    perturbed_features.append(extract_features(net, [(perturbed_data, labels)]))

perturbed_features = torch.cat(perturbed_features)
perturbed_features_np = perturbed_features.numpy()

# 扰动后特征空间可视化
features_perturbed_pca = pca.fit_transform(perturbed_features_np)
plt.figure(figsize=(8, 6))
plt.scatter(features_perturbed_pca[:, 0], features_perturbed_pca[:, 1], s=10, cmap='coolwarm')
plt.title('Feature Space after Perturbation')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid()
plt.savefig('feature_space_after.png')
plt.close()
