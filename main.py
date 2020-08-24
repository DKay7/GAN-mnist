from net import NetworkStuff
import torch

torch.cuda.empty_cache()
torch.manual_seed(97)


ns = NetworkStuff(file_name='gan-mnist-2', num_epochs=1000)

ns.load_model()
ns.train(start=97)
ns.make_gif()
ns.show_samples(show=True)
