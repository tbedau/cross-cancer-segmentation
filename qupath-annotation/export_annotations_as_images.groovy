// Script to export image regions corresponding to "Tumor" or "Normal" annotations

// Retrieve current image data and its name without extension
def imageData = getCurrentImageData()
def imageName = GeneralTools.stripExtension(imageData.getServer().getMetadata().getName())
def mpp = getCurrentServer().getPixelCalibration().getPixelWidthMicrons().round(3)

// Define and create the output directory within the project directory
def pathOutput = buildFilePath(PROJECT_BASE_DIR, 'image_export')
mkdirs(pathOutput)

// Set the downsampling rate
double downsample = 1

// Retrieve annotations labeled as "Tumor" or "Normal"
def annotations = getAnnotationObjects().findAll {
    it.getPathClass() && (it.getPathClass().getName().equalsIgnoreCase("Tumor") || it.getPathClass().getName().equalsIgnoreCase("Normal"))
}

// Process and export the image region for each annotation
for (annotation in annotations) {
    // Retrieve class name and create downsampled region request
    def className = annotation.getPathClass().getName().toLowerCase()
    def region = RegionRequest.createInstance(imageData.getServer().getPath(), downsample, annotation.getROI())

    // Construct the output path and export the image region
    def individualPathOutput = buildFilePath(pathOutput, "${imageName}_${mpp}_${className}_original.jpg")
    writeImageRegion(imageData.getServer(), region, individualPathOutput)
}
