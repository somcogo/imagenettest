from training import TinyImageNetTrainingApp
from site_training import MultiSiteTrainingApp

exp1 = TinyImageNetTrainingApp(epochs=1000, batch_size=4096, logdir='whole', lr=1e-3, comment='lr3')
exp1.main()
exp2 = TinyImageNetTrainingApp(epochs=1000, batch_size=4096, logdir='whole', lr=1e-4, comment='lr4')
exp2.main()
exp3 = TinyImageNetTrainingApp(epochs=1000, batch_size=4096, logdir='whole', lr=1e-5, comment='lr5')
exp3.main()
exp4 = TinyImageNetTrainingApp(epochs=1000, batch_size=4096, logdir='site1', lr=1e-3, site=0, comment='lr3')
exp4.main()
exp5 = TinyImageNetTrainingApp(epochs=1000, batch_size=4096, logdir='site1', lr=1e-4, site=0, comment='lr4')
exp5.main()
exp6 = TinyImageNetTrainingApp(epochs=1000, batch_size=4096, logdir='site1', lr=1e-5, site=0, comment='lr5')
exp6.main()

# exp2 = MultiSiteTrainingApp(epochs=10, batch_size=4096, logdir='test', lr=1e-4, comment='test')
# exp2.main()

# exp2 = MultiSiteTrainingApp(epochs=10, batch_size=4096, logdir='test', lr=1e-4, comment='test', layer='layer1')
# exp2.main()