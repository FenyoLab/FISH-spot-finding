
from skimage import io
import scyjava_config
scyjava_config.add_options('-Xmx12g') # allow 12g of memory to JVM
import imagej
ij = imagej.init('/Applications/Fiji.app', ) #headless=False) # load Fiji from local installation
#print(ij.getApp().getInfo(True))
from jnius import autoclass
WindowManager = autoclass('ij.WindowManager')

base_dir = "/Users/sarahkeegan/Dropbox/mac_files/fenyolab/data_and_results/davoli/Beautiful FISH/nuclei/"
file_name = "6_nucl_12_1.tif"
img1 = io.imread(base_dir + '/' + "6_nucl_12_1.tif")

ij.io().open(base_dir + '/' + "6_nucl_12_1.tif")
ij.py.run_macro("""run("Find Maxima...", "prominence=50 output=[Single Points]");""")
result = WindowManager.getCurrentImage()
result_array = ij.py.from_java(result)

# from jnius import autoclass,cast
# MaximumFinder = autoclass('ij.plugin.filter.MaximumFinder')
# #mf = cast(MaximumFinder, ij.get('ij.plugin.filter.MaximumFinder'))
#
#
# LegacyService = autoclass('net.imagej.legacy.LegacyService')
# legacyService = cast(LegacyService, ij.get("net.imagej.legacy.LegacyService"))
# ij.legacy_enabled = legacyService.isActive()
# print(ij.legacy_enabled)
#
# img1 = ij.io().open(base_dir + '/' + "6_nucl_12_1.tif")
# ij.script().run("Find Maxima...", "prominence=50 output=[Single Points]");
# #print(ij.script().help('run'))
#
#
# #WindowManager = autoclass('net.imagej.legacy.command.LegacyCommandFinder')
# # cmdFinder = autoclass('net.imagej.legacy.command.LegacyCommandFinder')
# # print(cmdFinder.findCommands())
# #
# #
# # imp = IJ.openImage("/Users/sarahkeegan/Dropbox/mac_files/fenyolab/data_and_results/davoli/Beautiful FISH/nuclei/6_nucl_2_0.tif");
# # IJ.run(imp, "Find Maxima...", "prominence=50 output=[Single Points]");
# # #
# #
#
# ex_list = [1, 2, 3, 4]
# print(type(ex_list))
# java_list = ij.py.to_java(ex_list)
# print(type(java_list))
#
# img1 = ij.io().open(base_dir + '/' + "6_nucl_12_1.tif")
# print(type(img1))
#
# macro = """
# #@ String file_name
# #@ int prom
# #@ String out_file_name
# open("/Users/sarahkeegan/Dropbox/mac_files/fenyolab/data_and_results/davoli/Beautiful FISH/nuclei/6_nucl_12_1.tif");
# run("Find Maxima...", "prominence=50 output=[Single Points]");
# saveAs("Tiff", "/Users/sarahkeegan/Dropbox/mac_files/fenyolab/data_and_results/davoli/Beautiful FISH/nuclei/6_nucl_12_1.tif Maxima.tif");
# """
# args = {
#     'file_name': 'temp',
#     'prom': 50,
#     'out_file_name': 'temp'
# }
# result = ij.py.run_macro(macro, args)

#print(result.getOutput('greeting'))

#ij.IJ().run(img1,"Find Maxima...","noise=100 output=[Single Points] exclude light") #for example

#ij.py.show(img1, cmap = 'gray')

#img1 = io.imread(base_dir + '/' + "6_nucl_12_1.tif")
#img2 = io.imread(base_dir + '/' + "6_nucl_12_1.tif")
#img_out = ij.py.new_numpy_image(img1)

#ip = ij.IJ.openImage(base_dir + '/' + "6_nucl_12_1.tif")
#ij.run(F1,"Find Maxima...","noise=100 output=[Single Points] exclude light") #for example
#ij.op().run('multiply', ij.py.to_java(img_out), ij.py.to_java(img1), ij.py.to_java(img2))

#io.imsave(base_dir + '/' + "result.tif", img_out)
#ij.run(F1,"Find Maxima...","noise=100 output=[Single Points] exclude light") #for example
#POINTs=WM.getCurrentImage()


# args2={'prominence':'50',
#       'output':'[Single Points]'}
# ij.py.run_plugin("Find Maxima...", args2)
# img1 = ij.io().open(base_dir + '/' + "6_nucl_12_1.tif")
# imp = ij.py.get_image_plus()
# print(type(imp))


