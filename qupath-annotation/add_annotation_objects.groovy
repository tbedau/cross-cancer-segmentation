import qupath.lib.objects.PathObjects
import qupath.lib.roi.ROIs
import qupath.lib.regions.ImagePlane

// Define the size of the annotation rectangles (in pixels at full resolution)
int rectWidth = 12500
int rectHeight = 10000

// Define the annotation classes using QuPath's built-in method
def classTumor = getPathClass("Tumor")
def classNormal = getPathClass("Normal")

// Get the current image data (assumes a single image is selected)
def imageData = getCurrentImageData()
def entry = getProjectEntry()
if (imageData == null) {
    print("No image is open.")
    return
}

// Define the plane
int z = 0
int t = 0
def plane = ImagePlane.getPlane(z, t)

// Calculate the center position for the annotations
def server = imageData.getServer()
int centerX = server.getWidth() / 2
int centerY = server.getHeight() / 2

// Calculate the top-left coordinates for the rectangles
int x1 = centerX - rectWidth / 2
int y1 = centerY - rectHeight / 2

// Create and add the tumor annotation
def roiTumor = ROIs.createRectangleROI(x1, y1, rectWidth, rectHeight, plane)
def annotationTumor = PathObjects.createAnnotationObject(roiTumor, classTumor)
imageData.getHierarchy().addObject(annotationTumor)

// Create and add the normal annotation
// Adjust the xOffset to avoid overlap
int xOffset = rectWidth + 1000 // Change this value if more offset is needed
def roiNormal = ROIs.createRectangleROI(x1 + xOffset, y1, rectWidth, rectHeight, plane)
def annotationNormal = PathObjects.createAnnotationObject(roiNormal, classNormal)
imageData.getHierarchy().addObject(annotationNormal)

// Save changes to the image data
entry.saveImageData(imageData)
