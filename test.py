from net import NetworkStuff  # , Generator1, Discriminator1
import torch

torch.cuda.empty_cache()
torch.manual_seed(97)

# gen1 = Generator1()
# dis1 = Discriminator1()

ns = NetworkStuff(file_name='gan-mnist-2', num_epochs=100)

ns.load_model()
# ns.train(start=50)
# ns.make_gif(fps=42)
ns.show_samples(show=True)
