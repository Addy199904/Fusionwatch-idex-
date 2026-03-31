# map_engine.py

def build_tactical_map_html(centre_lat: float = 17.6868, centre_lon: float = 83.2185, zoom: int = 14, theme: dict = None) -> str:
    """Generates the HTML/JS for the Leaflet map with Geofencing, Basemap Switching, and Tactical Styling."""
    
    if theme is None:
        theme = {
            "BG_DARK": "#0a0e1a", "BG_PANEL": "#0d1220", "BORDER": "#1e2a40",
            "ACCENT": "#38bdf8", "AMBER": "#f39c12", "RED": "#e74c3c", "GREEN": "#2ecc71",
            "TEXT_PRI": "#c8d0e0", "TEXT_MUT": "#3a4a60"
        }

    return f"""<!DOCTYPE html>
<html>
<head>
<link rel='stylesheet' href='https://unpkg.com/leaflet@1.9.4/dist/leaflet.css'/>
<script src='https://unpkg.com/leaflet@1.9.4/dist/leaflet.js'></script>

<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet.draw/1.0.4/leaflet.draw.css" />
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet.draw/1.0.4/leaflet.draw.js"></script>

<style>
  html, body, #map {{ margin:0; padding:0; width:100%; height:100%; background-color: {theme['BG_DARK']}; }}
  .leaflet-popup-content {{ font-family: 'Courier New', monospace; font-size: 13px; font-weight: bold; }}
  .leaflet-draw-toolbar a {{ background-color: {theme['BG_PANEL']}; border-color: {theme['BORDER']}; color: {theme['ACCENT']}; }}
  .leaflet-draw-toolbar a:hover {{ background-color: {theme['BORDER']}; }}
  
  /* Styling for the new layer control box to match tactical theme */
  .leaflet-control-layers {{ background: {theme['BG_PANEL']} !important; color: {theme['TEXT_PRI']}; border: 1px solid {theme['BORDER']} !important; border-radius: 4px; }}
  .leaflet-control-layers-expanded {{ color: {theme['TEXT_PRI']}; font-family: 'Segoe UI', sans-serif; font-size: 12px; font-weight: bold; }}
</style>
</head>
<body>
<div id='map'></div>
<script>
// --- BASEMAP PROVIDERS ---
var cartoDark = L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{ 
    maxZoom: 19, attribution: '&copy; CARTO' 
}});

var googleSat = L.tileLayer('https://{{s}}.google.com/vt/lyrs=s&x={{x}}&y={{y}}&z={{z}}', {{
    maxZoom: 20, subdomains:['mt0','mt1','mt2','mt3'], attribution: '&copy; Google'
}});

var esriSat = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
    maxZoom: 19, attribution: 'Tiles &copy; Esri'
}});

var osm = L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
    maxZoom: 19, attribution: '&copy; OpenStreetMap'
}});

// Initialize map with Tactical Dark as default
var map = L.map('map', {{
    center: [{centre_lat}, {centre_lon}],
    zoom: {zoom},
    layers: [cartoDark]
}});

// Add the Basemap Switcher Control
var baseMaps = {{
    "Tactical Dark (Demo Safe)": cartoDark,
    "Google Satellite (Test)": googleSat,
    "ESRI Satellite (Test)": esriSat,
    "Street Map (Test)": osm
}};
L.control.layers(baseMaps, null, {{position: 'topright'}}).addTo(map);

// --- AOI GEOFENCE DRAWING LOGIC ---
var drawnItems = new L.FeatureGroup();
map.addLayer(drawnItems);

var drawControl = new L.Control.Draw({{
    draw: {{
        polygon: {{
            allowIntersection: false,
            showArea: true,
            shapeOptions: {{ color: '{theme['ACCENT']}', weight: 2, dashArray: '5, 5', fillOpacity: 0.1 }}
        }},
        rectangle: {{
            shapeOptions: {{ color: '{theme['AMBER']}', weight: 2, dashArray: '5, 5', fillOpacity: 0.1 }}
        }},
        polyline: false, circle: false, marker: false, circlemarker: false
    }},
    edit: {{ featureGroup: drawnItems }}
}});
map.addControl(drawControl);

// Math extraction and Python bridge
function updateAOIData() {{
    var layers = drawnItems.getLayers();
    if (layers.length === 0) {{
        document.title = "AOI_CLEAR";
        return;
    }}
    
    var layer = layers[0];
    var areaSqKm = 0;
    var coordsStr = [];

    // Extract exact vertices and calculate geodesic area
    if (layer instanceof L.Polygon || layer instanceof L.Rectangle) {{
        var latlngs = layer.getLatLngs()[0];
        var area = L.GeometryUtil.geodesicArea(latlngs);
        areaSqKm = (area / 1000000).toFixed(3); // Convert Sq Meters to Sq KM
        
        latlngs.forEach(function(ll) {{
            coordsStr.push(Math.abs(ll.lat).toFixed(4) + (ll.lat >= 0 ? "N" : "S") + " " + Math.abs(ll.lng).toFixed(4) + (ll.lng >= 0 ? "E" : "W"));
        }});
    }}

    var payload = {{
        area: areaSqKm,
        coords: coordsStr.join("  |  ")
    }};
    
    // Send data to Python via hidden page title
    document.title = "AOI_UPDATE|" + JSON.stringify(payload);
}}

map.on(L.Draw.Event.CREATED, function (event) {{
    drawnItems.clearLayers(); // Enforce a single AOI for exact telemetry
    drawnItems.addLayer(event.layer);
    updateAOIData();
}});

map.on(L.Draw.Event.EDITED, updateAOIData);
map.on(L.Draw.Event.DELETED, updateAOIData);
// -----------------------------------

var imageOverlays = {{}};
window.overlayGeoTIFF = function(layerId, base64Data, minLat, minLon, maxLat, maxLon) {{
    var imageUrl = "data:image/png;base64," + base64Data;
    var imageBounds = [[minLat, minLon], [maxLat, maxLon]];
    var overlay = L.imageOverlay(imageUrl, imageBounds, {{opacity: 0.8}}).addTo(map);
    imageOverlays[layerId] = overlay;
    map.fitBounds(imageBounds); 
}};

window.toggleLayer = function(layerId, isVisible) {{
    if (imageOverlays[layerId]) {{
        if (isVisible) map.addLayer(imageOverlays[layerId]);
        else map.removeLayer(imageOverlays[layerId]);
    }}
}};

window.clearOverlays = function() {{ 
    for (var id in imageOverlays) {{ map.removeLayer(imageOverlays[id]); }}
    imageOverlays = {{}}; 
}};

var markers = [];
window.addSingleDetection = function(lat, lon, cls, risk, tgtId) {{
    var colours = {{ 'CRITICAL': '{theme['RED']}', 'HIGH': '{theme['AMBER']}', 'MEDIUM': '#eab308', 'LOW': '{theme['GREEN']}' }};
    var m = L.circleMarker([lat, lon], {{
        radius: 7, fillColor: colours[risk], color: '#000000', weight: 1, opacity: 1, fillOpacity: 0.9
    }}).addTo(map);
    m.bindPopup(tgtId + "<br>" + cls + " [" + risk + "]");
    markers.push(m);
}};

window.panTo = function(lat, lon) {{ map.setView([lat, lon], 16); }};
window.clearMarkers = function() {{ markers.forEach(m => map.removeLayer(m)); markers = []; }};

window.getGeofenceBounds = function() {{
    var layers = drawnItems.getLayers();
    if (layers.length === 0) return null;
    var bounds = layers[0].getBounds();
    return [bounds.getNorth(), bounds.getSouth(), bounds.getEast(), bounds.getWest()];
}};
</script>
</body>
</html>"""