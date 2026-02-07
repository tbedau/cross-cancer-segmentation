import qupath.lib.objects.PathObject
import qupath.lib.objects.PathAnnotationObject
import qupath.lib.roi.RectangleROI
import qupath.lib.regions.RegionRequest
import qupath.lib.images.servers.LabeledImageServer
import java.awt.image.BufferedImage
import javax.imageio.ImageIO
*// User parameters - adjust these according to your needs*
*// Class name of the rectangular bounding box to define the export region*
def boundingBoxClassName = "EXPORT_ROI"
*// Output resolution in pixels per micron (adjust based on your needs)*
double downsample = 1.0
*// Output directory path - change this to your desired location*
def outputDir = buildFilePath(PROJECT_BASE_DIR, 'ground_truth_class_maps')
mkdirs(outputDir)
*// Get the current image data*
def imageData = getCurrentImageData()
def hierarchy = imageData.getHierarchy()
def server = imageData.getServer()
*// Get the original image name (without extension)*
def imageName = server.getMetadata().getName()
if (imageName.contains(".")) {
 imageName = imageName.substring(0, imageName.lastIndexOf('.'))
}
*// Find all annotations with the bounding box class*
def boundingBoxes = hierarchy.getAnnotationObjects().findAll { it.getPathClass() != null && it.getPathClass().getName() == boundingBoxClassName }
*// Check if we found any bounding boxes*
if (boundingBoxes.isEmpty()) {
print("No annotations with class '${boundingBoxClassName}' found!")
return
}
*// Create a labeled image server with minimal settings*
def labelServer = new LabeledImageServer.Builder(imageData)
 .backgroundLabel(0) *// Just using the basic method with only the integer value*
 .addLabel('TUMOR', 1)
 .addLabel('ADENOM_HG', 1)
 .addLabel('TU_STROMA', 2)
 .addLabel('MUC', 3)
 .addLabel('ADENOM_LG', 3)
 .addLabel('SUBMUC', 3)
 .addLabel('MUSC_PROP', 3)
 .addLabel('MUSC_MUC', 3)
 .addLabel('ADVENT', 4)
 .addLabel('VESSEL', 4)
 .addLabel('LYMPH_NODE', 5)
 .addLabel('LYMPH_TIS', 5)
 .addLabel('LYMPH_AGGR', 5)
 .addLabel('ULCUS', 6)
 .addLabel('NECROSIS', 6)
 .addLabel('BLOOD', 7)
 .addLabel('MUCIN', 8)
 .build()
*// Process each bounding box*
for (def i = 0; i < boundingBoxes.size(); i++) {
def boundingBox = boundingBoxes[i]
def roi = boundingBox.getROI()
*// Skip if not a rectangle*
if (!(roi instanceof RectangleROI)) {
print("Warning: Annotation #${i+1} with class '${boundingBoxClassName}' is not a rectangle, skipping...")
continue
 }
*// Get ROI coordinates for the filename*
int x = (int)roi.getBoundsX()
int y = (int)roi.getBoundsY()
int width = (int)roi.getBoundsWidth()
int height = (int)roi.getBoundsHeight()
*// Create a more descriptive filename*
def filename = "${imageName}_class_map_ground_truth_roi_${i+1}.png"
*// Create a region request for the bounding box*
def request = RegionRequest.createInstance(
 server.getPath(),
 downsample,
 roi
 )
*// Export as class map (PNG)*
def outputFile = new File(outputDir, filename)
def img = labelServer.readBufferedImage(request)
ImageIO.write(img, "PNG", outputFile)
print("Exported class map to ${outputFile.getAbsolutePath()}")
}
print("Export complete!")
