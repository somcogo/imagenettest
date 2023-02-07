from training import TinyImageNetTrainingApp
from site_training import MultiSiteTrainingApp
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# exp1 = TinyImageNetTrainingApp(epochs=500, batch_size=4096, logdir='whole', lr=1e-3, comment='lr3aug')
# exp1.main()
# exp2 = TinyImageNetTrainingApp(epochs=500, batch_size=4096, logdir='whole', lr=1e-4, comment='lr4aug')
# exp2.main()
# exp3 = TinyImageNetTrainingApp(epochs=1000, batch_size=4096, logdir='whole', lr=1e-5, comment='lr5')
# exp3.main()
exp4 = TinyImageNetTrainingApp(epochs=1000, batch_size=2048, logdir='site1', lr=1e-3, site=0, comment='lr3')
exp4.main()
exp5 = TinyImageNetTrainingApp(epochs=1000, batch_size=2048, logdir='site1', lr=1e-4, site=0, comment='lr4')
exp5.main()
exp6 = TinyImageNetTrainingApp(epochs=1000, batch_size=2048, logdir='site1', lr=1e-5, site=0, comment='lr5')
exp6.main()

# exp2 = MultiSiteTrainingApp(epochs=10, batch_size=4096, logdir='test', lr=1e-4, comment='test')
# exp2.main()

# exp2 = MultiSiteTrainingApp(epochs=500, batch_size=4096, logdir='layer4', lr=1e-4, comment='site1lr5-lr4layer4.1', layer='layer4', sub_layer='1')
# exp2.main()