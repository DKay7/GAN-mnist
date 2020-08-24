import os
import torch
import imageio
import torchvision
from torch import nn
from tqdm import tqdm
import torch.nn.functional as f
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(x.size(0), 784)
        output = self.model(x)
        return output


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 784),
            nn.Tanh(),
        )

    def forward(self, x):
        output = self.model(x)
        output = output.view(x.size(0), 1, 28, 28)
        return output


class NetworkStuff:
    def __init__(self, file_name, load=False, num_epochs=50,
                 gen=Generator(), dis=Discriminator()):

        self.batch_size = 32
        self.lr = 0.0001
        self.num_epochs = num_epochs
        self.file_name = file_name

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        if not load:
            self.discriminator = dis.to(self.device)
            self.generator = gen.to(self.device)
        else:
            self.load_model()

        self.criterion = nn.BCELoss()

        self.optimizer_discriminator = torch.optim.Adam(
            self.discriminator.parameters(),
            self.lr,
        )
        self.optimizer_generator = torch.optim.Adam(
            self.generator.parameters(),
            self.lr,
        )

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        self.train_set = torchvision.datasets.MNIST(
            root=os.path.abspath("data"), train=True, download=True,
            transform=self.transform
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_set, batch_size=self.batch_size,
            shuffle=True
        )

        self.fixed_samples = torch.randn((self.batch_size, 100)).to(self.device)

    def train(self, start=0):
        print('Training will compute on: ', self.device)

        real_samples_labels = torch.ones((self.batch_size, 1)).to(self.device)
        generated_samples_labels = torch.zeros((self.batch_size, 1)).to(self.device)
        all_samples_labels = torch.cat((real_samples_labels, generated_samples_labels))

        for epoch in tqdm(range(start, self.num_epochs), initial=start, total=self.num_epochs):
            for n, (real_samples, mnist_labels) in enumerate(self.train_loader):

                # Данные для тренировки дискриминатора
                real_samples = real_samples.to(self.device)
                latent_space_samples = torch.randn((self.batch_size, 100)).to(self.device)
                generated_samples = self.generator(latent_space_samples)
                all_samples = torch.cat((real_samples, generated_samples))

                # Обучение дискриминатора
                self.discriminator.zero_grad()
                output_discriminator = self.discriminator(all_samples.detach())
                loss_discriminator = self.criterion(output_discriminator, all_samples_labels)
                loss_discriminator.backward()
                self.optimizer_discriminator.step()

                # Данные для обучения генератора
                # latent_space_samples = torch.randn((self.batch_size, 100)).to(self.device)

                # Обучение генератора
                self.generator.zero_grad()
                # generated_samples = self.generator(latent_space_samples)
                output_discriminator_generated = self.discriminator(generated_samples)
                loss_generator = self.criterion(output_discriminator_generated, real_samples_labels)
                loss_generator.backward()
                self.optimizer_generator.step()

                if n == self.batch_size - 1 or n % 8 == 0:
                    tqdm.write(f"Epoch: {epoch} Loss D.: {loss_discriminator}\n" +
                               f"Epoch: {epoch} Loss G.: {loss_generator}")

            if epoch % 5 == 0:
                self.save_model()

            self.show_samples(file_name=str(epoch), data='fixed')

    def save_model(self, file_name=None):
        """
        Сохраняет параметры нейронной сети

        :param file_name: Имя файла для весов модели,
            если не передано, будет взято имя, переданное в конструктор класса
        """

        if file_name is None:
            file_name = self.file_name

        gen_path = 'models/' + file_name + '-generator' + '.pth'
        dis_path = 'models/' + file_name + '-discriminator' + '.pth'

        torch.save(self.generator.state_dict(), gen_path)
        torch.save(self.discriminator.state_dict(), dis_path)

    def load_model(self, file_name=None):
        """
        Загружает параметры нейронной сети

        :param file_name: Имя файла для весов модели,
            если не передано, будет взято имя, переданное в конструктор класса
        """

        if file_name is None:
            file_name = self.file_name

        gen_path = 'models/' + file_name + '-generator' + '.pth'
        dis_path = 'models/' + file_name + '-discriminator' + '.pth'

        self.generator.load_state_dict(torch.load(gen_path))
        self.discriminator.load_state_dict(torch.load(dis_path))

        self.generator = self.generator.to(self.device)
        self.discriminator = self.discriminator.to(self.device)

    def show_samples(self, show=False, file_name=None, data='random'):

        if data == 'random':
            latent_samples = torch.randn((self.batch_size, 100)).to(self.device)

        elif data == 'fixed':
            latent_samples = self.fixed_samples

        else:
            raise NameError

        with torch.no_grad():
            generated_samples = self.generator(latent_samples).cpu().detach()

        plt.suptitle(f'generated after {file_name} epochs')
        for i in range(16):
            plt.subplot(4, 4, i + 1)
            plt.imshow(generated_samples[i].reshape(28, 28), cmap="gray_r")
            plt.xticks([])
            plt.yticks([])

        if file_name is None:
            file_name = self.file_name

        if not os.path.isdir(os.path.join('gan_work_images', self.file_name)):
            os.mkdir(os.path.join('gan_work_images', self.file_name))

        plt.savefig(os.path.join('gan_work_images', self.file_name, file_name + '.png'))

        if show:
            plt.show()

    def make_gif(self, fps=24):

        all_pics_filenames = []

        for folder_data in os.walk(os.path.join('gan_work_images', self.file_name)):
            all_pics_filenames = sorted(folder_data[2], key=lambda x: int(x.split('.')[0]))

        with imageio.get_writer(os.path.join('gifs', self.file_name+'.gif'),
                                mode='I', fps=fps) as writer:

            for filename in tqdm(all_pics_filenames):
                image = imageio.imread(os.path.join('gan_work_images', self.file_name, filename))
                writer.append_data(image)
