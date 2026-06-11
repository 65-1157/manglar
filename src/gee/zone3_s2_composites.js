// ============================================================
// MANGLAR — Zone 3 Sentinel-2 Monthly Composites
// Role: Calibration baseline (near-pristine reference mangrove)
// Location: Baixada Maranhense
// Output: Monthly median composites 2017-2024
//         Bands: NDVI, EVI, CIre, NDWI + QA
// Export destination: Google Drive / MANGLAR_GEE_EXPORTS
// ============================================================

// ---- 1. Study area -----------------------------------------
var zone3 = ee.Geometry.Rectangle([-45.5, -3.5, -44.5, -2.8]);

// ---- 2. Global Mangrove Watch mask (2020 extent) -----------
var gmw = ee.ImageCollection('projects/sat-io/open-datasets/GMW/annual')
  .filter(ee.Filter.calendarRange(2020, 2020, 'year'))
  .first()
  .select('b1')
  .clip(zone3);

var mangrove_mask = gmw.eq(1);

// ---- 3. Cloud masking function (SCL band, L2A) -------------
function maskS2clouds(image) {
  var scl = image.select('SCL');
  // Keep: vegetation(4), bare soil(5), water(6), unclassified(7)
  // Exclude: cloud shadow(3), cloud medium(8), cloud high(9), cirrus(10)
  var clear = scl.neq(3)
    .and(scl.neq(8))
    .and(scl.neq(9))
    .and(scl.neq(10));
  return image.updateMask(clear)
    .divide(10000)  // Scale to reflectance
    .copyProperties(image, ['system:time_start']);
}

// ---- 4. Spectral index computation -------------------------
function addIndices(image) {
  var ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI');
  var evi  = image.expression(
    '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
    { NIR: image.select('B8'),
      RED: image.select('B4'),
      BLUE: image.select('B2') }
  ).rename('EVI');
  var cire = image.expression(
    '(B7 / B5) - 1',
    { B7: image.select('B7'),
      B5: image.select('B5') }
  ).rename('CIre');
  var ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI');
  return image.addBands([ndvi, evi, cire, ndwi]);
}

// ---- 5. Monthly composite generator ------------------------
var startYear = 2017;
var endYear   = 2024;

// Build list of year-month combinations
var months = ee.List.sequence(1, 12);
var years  = ee.List.sequence(startYear, endYear);

var monthlyComposites = ee.ImageCollection(
  years.map(function(y) {
    return months.map(function(m) {
      var start = ee.Date.fromYMD(y, m, 1);
      var end   = start.advance(1, 'month');

      var composite = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(zone3)
        .filterDate(start, end)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
        .map(maskS2clouds)
        .map(addIndices)
        .select(['NDVI', 'EVI', 'CIre', 'NDWI'])
        .median()
        .updateMask(mangrove_mask)
        .set('year', y)
        .set('month', m)
        .set('system:time_start', start.millis());

      return composite.clip(zone3);
    });
  }).flatten()
);

print('Total monthly composites:', monthlyComposites.size());
print('First image:', monthlyComposites.first());

// ---- 6. Visualise NDVI for sanity check --------------------
var ndviVis = { min: 0.2, max: 0.9, palette: ['white','yellow','green','darkgreen'] };
Map.centerObject(zone3, 10);
Map.addLayer(
  monthlyComposites.filter(ee.Filter.eq('month', 7))
                   .filter(ee.Filter.eq('year', 2020))
                   .first()
                   .select('NDVI'),
  ndviVis,
  'NDVI July 2020 - Zone 3'
);
Map.addLayer(zone3, {color: 'red'}, 'Zone 3 boundary');

// ---- 7. Export to Google Drive -----------------------------
// Export each year as a multi-band GeoTIFF stack
// Band order: NDVI_YYYY_MM, EVI_YYYY_MM, CIre_YYYY_MM, NDWI_YYYY_MM
// One export task per year to stay within GEE memory limits

years.evaluate(function(yearList) {
  yearList.forEach(function(y) {

    // Stack all 12 months for this year into one image
    var yearStack = ee.ImageCollection(
      months.map(function(m) {
        var img = monthlyComposites
          .filter(ee.Filter.eq('year', y))
          .filter(ee.Filter.eq('month', m))
          .first();
        // Rename bands to include year and month
        var bandNames = ['NDVI','EVI','CIre','NDWI'].map(function(b) {
          return b + '_' + y + '_' + (m < 10 ? '0' + m : m);
        });
        return img.rename(bandNames);
      })
    ).toBands();

    Export.image.toDrive({
      image: yearStack,
      description: 'manglar_zone3_s2_' + y,
      folder: 'MANGLAR_GEE_EXPORTS',
      fileNamePrefix: 'zone3_s2_' + y,
      region: zone3,
      scale: 10,
      crs: 'EPSG:4326',
      maxPixels: 1e13,
      fileFormat: 'GeoTIFF'
    });

    print('Export task queued for year:', y);
  });
});
