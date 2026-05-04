// main.js — Metro-style rendering (líneas como trazos + burbujas por parada)

// ============================================================
// 1. Inicializar Mapa
// ============================================================
const map = L.map('map', {
    center: [40.7128, -74.0060],
    zoom: 12,
    minZoom: 11,
    zoomControl: false
});

const TILE_LIGHT = 'https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png';
const TILE_DARK  = 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png';
const TILE_OPTS  = { attribution: '&copy; OpenStreetMap contributors &copy; CARTO', subdomains: 'abcd', maxZoom: 20 };

let tileLayer = L.tileLayer(TILE_LIGHT, TILE_OPTS).addTo(map);

// Pane de estaciones por debajo de los trenes
map.createPane('stationPane');
map.getPane('stationPane').style.zIndex = 420;
map.getPane('stationPane').style.pointerEvents = 'none';

// Pane de trenes siempre encima
map.createPane('trainPane');
map.getPane('trainPane').style.zIndex = 620;

// ============================================================
// 2. Paleta oficial MTA
// ============================================================
const ROUTE_COLORS = {
    '1': '#EE352E', '2': '#EE352E', '3': '#EE352E',
    '4': '#00933C', '5': '#00933C', '6': '#00933C',
    '7': '#B933AD',
    'A': '#0039A6', 'C': '#0039A6', 'E': '#0039A6',
    'B': '#FF6319', 'D': '#FF6319', 'F': '#FF6319', 'M': '#FF6319',
    'G': '#6CBE45',
    'J': '#996633', 'Z': '#996633',
    'L': '#A7A9AC',
    'N': '#FCCC0A', 'Q': '#FCCC0A', 'R': '#FCCC0A', 'W': '#FCCC0A',
    'S': '#808183', 'SIR': '#0039A6'
};

// Separación fija en metros entre líneas paralelas (independiente del zoom)
const OFFSET_METERS = 15;

function getStationColor(routes) {
    if (!routes) return '#3B82F6';
    const primaryRoute = routes.split(' ')[0];
    return ROUTE_COLORS[primaryRoute] || '#3B82F6';
}

// ============================================================
// 3. Estado global
// ============================================================
let markersMap = {};
let allStations = [];
let stationsById = {};     // stationId -> station object
let currentAlerts = [];    // AlertPrediction[] del WebSocket (por route_id)
let routePolylines = {};   // lineCode -> L.polyline
let borderPolylines = [];  // líneas de borde oscuro (para limpiarlas al redibujar)
let stationRouteIndex = {}; // stationId -> [lineCode, lineCode, ...]

const SUBWAY_STATIONS_URL = '/api/stations';
const SUBWAY_ROUTES_URL   = '/api/routes';  // Nuevo endpoint: devuelve { lineCode: [stationId, ...] }

// ============================================================
// 4. Helpers de offset visual (para separar líneas paralelas)
// ============================================================

/**
 * Posición de cada línea dentro de su trunk line.
 * idx: índice 0-based dentro del grupo
 * total: número de rutas en el grupo
 * El offset centrado es: (idx - (total-1)/2) * OFFSET_METERS
 */
const ROUTE_PARALLEL_INDEX = {
    // IRT Broadway-7th Ave
    // Ruta 2 usa offsetOverride fijo: comparte vía con la 5 en Brooklyn (Eastern Pkwy) y necesita
    // suficiente separación para verse como línea roja independiente junto a la verde.
    '1': { idx: 0, total: 3 }, '2': { idx: 1, total: 3, offsetOverride: -30 }, '3': { idx: 2, total: 3 },
    // IRT Lexington Ave
    // Ruta 5 idem: offsetOverride opuesto al de la 2 → 60 m de separación en Brooklyn.
    '4': { idx: 0, total: 3 }, '5': { idx: 1, total: 3, offsetOverride: +30 }, '6': { idx: 2, total: 3 },
    // IND 8th Ave
    'A': { idx: 0, total: 3 }, 'C': { idx: 1, total: 3 }, 'E': { idx: 2, total: 3 },
    // IND 6th Ave
    'B': { idx: 0, total: 4 }, 'D': { idx: 1, total: 4 }, 'F': { idx: 2, total: 4 }, 'M': { idx: 3, total: 4 },
    // BMT Broadway
    'N': { idx: 0, total: 4 }, 'Q': { idx: 1, total: 4 }, 'R': { idx: 2, total: 4 }, 'W': { idx: 3, total: 4 },
    // BMT Nassau
    'J': { idx: 0, total: 2 }, 'Z': { idx: 1, total: 2 },
    // Single-trunk routes
    '7':   { idx: 0, total: 1 },
    'G':   { idx: 0, total: 1 },
    'L':   { idx: 0, total: 1 },
    'S':   { idx: 0, total: 1 },
    'GS':  { idx: 0, total: 1 },
    'FS':  { idx: 0, total: 1 },
    'H':   { idx: 0, total: 1 },
    'SIR': { idx: 0, total: 1 },
};

/**
 * Aplica un offset perpendicular fijo en metros a una lista de puntos [[lat, lon], ...].
 * Independiente del zoom — las líneas no saltan al hacer zoom.
 */
function offsetLatLngsMeters(points, offsetMeters) {
    if (offsetMeters === 0 || points.length < 2) return points;

    return points.map((pt, i) => {
        const prev = points[Math.max(0, i - 1)];
        const next = points[Math.min(points.length - 1, i + 1)];

        const cosLat = Math.cos(pt[0] * Math.PI / 180);
        const dlatM  = (next[0] - prev[0]) * 111320;
        const dlonM  = (next[1] - prev[1]) * 111320 * cosLat;
        const len    = Math.sqrt(dlatM * dlatM + dlonM * dlonM) || 1;

        const nx = -dlonM / len;
        const ny =  dlatM / len;

        return [
            pt[0] + nx * offsetMeters / 111320,
            pt[1] + ny * offsetMeters / (111320 * cosLat)
        ];
    });
}

// ============================================================
// 5. Crear marcador SVG estilo metro con burbujas por línea
// ============================================================

// Radio de burbuja según nivel de zoom
function markerRadiusForZoom(zoom) {
    if (zoom >= 16) return 10;
    if (zoom >= 14) return 8;
    if (zoom >= 13) return 6;
    if (zoom >= 12) return 4;
    return 3;
}

/**
 * Genera un L.divIcon con una o varias burbujas coloreadas
 * (una por línea que pasa por la parada), alineadas horizontalmente.
 */
function createMetroMarker(routes, zoom) {
    const routeList = routes ? routes.split(' ') : [];
    const R = markerRadiusForZoom(zoom ?? map.getZoom());
    const GAP = Math.max(1, Math.round(R * 0.25));
    const n = routeList.length;
    const totalW = n * (R * 2) + (n - 1) * GAP;
    const totalH = R * 2 + 4;

    const circles = routeList.map((route, i) => {
        const color = ROUTE_COLORS[route] || '#3B82F6';
        const cx = i * (R * 2 + GAP) + R;
        const cy = R + 2;
        const fontSize = Math.max(5, R - 1);
        // Ocultar letra si el radio es muy pequeño
        const label = R >= 5 ? (route.length <= 2 ? route : '') : '';
        return `<circle cx="${cx}" cy="${cy}" r="${R}"
                    fill="${color}"
                    stroke="white" stroke-width="${R >= 6 ? 1.5 : 1}"
                    style="filter:drop-shadow(0 0 3px ${color}88)"/>
                <text x="${cx}" y="${cy}"
                    dominant-baseline="central" text-anchor="middle"
                    font-family="Arial,sans-serif" font-size="${fontSize}"
                    font-weight="bold" fill="white">${label}</text>`;
    }).join('');

    const svg = `<svg xmlns="http://www.w3.org/2000/svg"
                    width="${totalW}" height="${totalH}"
                    viewBox="0 0 ${totalW} ${totalH}"
                    style="overflow:visible">
                    ${circles}
                </svg>`;

    return L.divIcon({
        className: 'metro-marker',
        html: `<div style="pointer-events:auto">${svg}</div>`,
        iconSize: [totalW, totalH],
        iconAnchor: [totalW / 2, totalH / 2]
    });
}

// ============================================================
// 6. Dibujar líneas de metro como polilíneas
// ============================================================

/**
 * El backend devuelve los IDs en el orden oficial de la línea (via _ROUTE_ORDER).
 * Construimos segmentos continuos cortando:
 *   a) cuando una estación no se encuentra en el CSV, o
 *   b) cuando la distancia entre dos paradas consecutivas supera MAX_SEGMENT_KM
 *      (indica que hay paradas intermedias no resueltas → evita líneas diagonales largas).
 */
const MAX_SEGMENT_KM = 3.0;

function haversineKm(lat1, lon1, lat2, lon2) {
    const R = 6371;
    const dLat = (lat2 - lat1) * Math.PI / 180;
    const dLon = (lon2 - lon1) * Math.PI / 180;
    const a = Math.sin(dLat / 2) ** 2 +
              Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
              Math.sin(dLon / 2) ** 2;
    return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

function drawRouteLines(routeData, stationsById) {
    const lineList = Object.keys(routeData);

    lineList.forEach((lineCode) => {
        const rawIds = routeData[lineCode];
        const color = ROUTE_COLORS[lineCode] || '#3B82F6';

        // Construir lista de segmentos continuos con doble corte:
        // (1) estación no encontrada, (2) salto geográfico > MAX_SEGMENT_KM
        const segments = [];
        let currentSegment = [];

        rawIds.forEach(id => {
            const st = stationsById[id];
            if (!st) {
                // Estación sin datos: cerrar segmento
                if (currentSegment.length >= 2) segments.push(currentSegment);
                currentSegment = [];
                return;
            }
            if (currentSegment.length > 0) {
                const prev = currentSegment[currentSegment.length - 1];
                const dist = haversineKm(prev[0], prev[1], st.lat, st.lon);
                if (dist > MAX_SEGMENT_KM) {
                    // Salto demasiado grande → hay paradas no resueltas entre medias
                    if (currentSegment.length >= 2) segments.push(currentSegment);
                    currentSegment = [];
                }
            }
            currentSegment.push([st.lat, st.lon]);
        });
        if (currentSegment.length >= 2) segments.push(currentSegment);

        // Dibujar cada segmento continuo
        segments.forEach(latlngs => {
            // Borde oscuro debajo para dar profundidad
            L.polyline(latlngs, {
                color: darkenColor(color, 0.4),
                weight: 7,
                opacity: 0.6,
                lineJoin: 'round',
                lineCap: 'round'
            }).addTo(map);

            // Línea principal
            L.polyline(latlngs, {
                color: color,
                weight: 4,
                opacity: 0.9,
                lineJoin: 'round',
                lineCap: 'round',
                className: `metro-line metro-line-${lineCode}`
            }).addTo(map);
        });

        // Guardar referencia al primer segmento para uso posterior
        if (segments.length > 0) {
            routePolylines[lineCode] = segments[0];
        }
    });
}

function darkenColor(hex, factor) {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgb(${Math.round(r * factor)},${Math.round(g * factor)},${Math.round(b * factor)})`;
}

// ============================================================
// 7. Carga de datos y renderizado
// ============================================================

/**
 * Dibuja las líneas usando la geometría oficial GTFS (shapes).
 * Cada ruta tiene un trazo lat/lon que sigue el carril real de la MTA.
 * No depende de matching de nombres de estación.
 */
let rawShapesDataCache = null;

function redrawShapes() {
    if (!rawShapesDataCache) return;
    
    // Limpiar polilíneas viejas
    Object.values(routePolylines).forEach(p => map.removeLayer(p));
    borderPolylines.forEach(p => map.removeLayer(p));
    routePolylines = {};
    borderPolylines = [];

    Object.entries(rawShapesDataCache).forEach(([routeCode, data]) => {
        const color = ROUTE_COLORS[routeCode] || '#3B82F6';

        const parallel = ROUTE_PARALLEL_INDEX[routeCode] || { idx: 0, total: 1 };
        const centerOffset = (parallel.total - 1) / 2;
        // offsetOverride permite asignar un offset fijo en metros (independiente de idx/total)
        // Útil cuando dos rutas comparten vía solo en algunas secciones (ej. 2 y 5 en Brooklyn).
        const offsetIndex = parallel.offsetOverride !== undefined
            ? parallel.offsetOverride / OFFSET_METERS
            : (parallel.idx - centerOffset);

        // Compatibilidad: el backend puede devolver [[lat,lon],...] o [[[lat,lon],...],...]
        const segmentsList = (data.length > 0 && Array.isArray(data[0][0])) ? data : [data];

        segmentsList.forEach((points, segIdx) => {
            const latlngs = offsetIndex !== 0 ? offsetLatLngsMeters(points, offsetIndex * OFFSET_METERS) : points;

            // Borde oscuro solo en el primer segmento
            if (offsetIndex === 0 && segIdx === 0) {
                const border = L.polyline(points, {
                    color: darkenColor(color, 0.4),
                    weight: 7,
                    opacity: 0.5,
                    lineJoin: 'round',
                    lineCap: 'round'
                }).addTo(map);
                borderPolylines.push(border);
            }

            const poly = L.polyline(latlngs, {
                color: color,
                weight: 3.5,
                opacity: 0.88,
                lineJoin: 'round',
                lineCap: 'round',
                className: `metro-line metro-line-${routeCode}`
            }).addTo(map);

            if (segIdx === 0) routePolylines[routeCode] = poly;
        });
    });
}

function drawShapeLines(shapesData) {
    rawShapesDataCache = shapesData;
    redrawShapes();
    
    // Ocultar líneas al empezar el zoom y redibujar al terminar (evita descuadre visual)
    map.on('zoomstart', () => {
        Object.values(routePolylines).forEach(p => {
            if (p && p.getElement) {
                const el = p.getElement();
                if (el) el.style.opacity = '0';
            }
        });
    });
    map.on('zoomend', () => {
        redrawShapes();
    });
    
    console.log(`Shapes GTFS dibujados para ${Object.keys(shapesData).length} rutas con offsets geográficos fijos.`);
}


Promise.all([
    fetch(SUBWAY_STATIONS_URL).then(r => r.json()),
    fetch('/api/shapes').then(r => r.json()).catch(() => ({})),
    fetch('/api/routes').then(r => r.json()).catch(() => ({})),
    fetch('/api/warmup').catch(() => null),
]).then(([stations, shapesData, routeData]) => {
    allStations = stations;

    stationsById = {};
    stations.forEach(st => {
        stationsById[st.id] = st;
        stationRouteIndex[st.id] = st.routes ? st.routes.split(' ') : [];
    });

    // GTFS shapes: geometría exacta del carril. Fallback: conexión entre paradas.
    if (Object.keys(shapesData).length > 0) {
        drawShapeLines(shapesData);
    } else {
        console.warn('GTFS shapes no disponibles, usando conexión entre estaciones.');
        drawRouteLines(routeData, stationsById);
    }

    // Dibujar marcadores encima
    stations.forEach(station => {
        const icon = createMetroMarker(station.routes);
        const marker = L.marker([station.lat, station.lon], {
            icon,
            pane: 'stationPane',
        });
        marker.on('click', () => openStationDetails(station));
        markersMap[station.id] = marker.addTo(map);
    });

    // Redibujar marcadores al cambiar zoom para escalar su tamaño
    function redrawMarkers() {
        const zoom = map.getZoom();
        allStations.forEach(station => {
            const marker = markersMap[station.id];
            if (marker) marker.setIcon(createMetroMarker(station.routes, zoom));
        });
    }
    map.on('zoomend', redrawMarkers);

    console.log(`${stations.length} estaciones renderizadas con estilo metro.`);
    initRoutePlanner();

    // Hide initial loading overlay
    const overlay = document.getElementById('loading-overlay');
    if (overlay) {
        overlay.classList.add('fade-out');
        setTimeout(() => overlay.remove(), 520);
    }
}).catch(err => {
    console.error("Error al cargar datos:", err);
    const overlay = document.getElementById('loading-overlay');
    if (overlay) {
        overlay.classList.add('fade-out');
        setTimeout(() => overlay.remove(), 520);
    }
});


// ============================================================
// 8. Posiciones de trenes en tiempo real
// ============================================================

const trainLayer = L.layerGroup().addTo(map);

function createTrainIcon(routeId, isPredictable = true) {
    const color = ROUTE_COLORS[routeId] || '#888888';
    const w = 14, h = 8;

    const svg = isPredictable
        ? `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 14 8">
            <rect x="0.5" y="0.5" width="11" height="5.5" rx="1.5" fill="${color}" fill-opacity="0.78" stroke="white" stroke-width="0.8"/>
            <rect x="1.5" y="1.5" width="3" height="2.5" rx="0.8" fill="white" fill-opacity="0.25"/>
            <rect x="5.5" y="1.5" width="2" height="2.5" rx="0.8" fill="white" fill-opacity="0.25"/>
            <rect x="8.5" y="1.5" width="2" height="2.5" rx="0.8" fill="white" fill-opacity="0.25"/>
            <circle cx="3"  cy="7" r="1" fill="#111" stroke="white" stroke-width="0.6"/>
            <circle cx="9" cy="7" r="1" fill="#111" stroke="white" stroke-width="0.6"/>
        </svg>`
        : `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 14 8" opacity="0.55">
            <rect x="0.5" y="0.5" width="11" height="5.5" rx="1.5" fill="#aaa" fill-opacity="0.35" stroke="${color}" stroke-width="1" stroke-dasharray="2.5,1.5"/>
            <rect x="1.5" y="1.5" width="3" height="2.5" rx="0.8" fill="${color}" fill-opacity="0.18"/>
            <rect x="5.5" y="1.5" width="2" height="2.5" rx="0.8" fill="${color}" fill-opacity="0.18"/>
            <rect x="8.5" y="1.5" width="2" height="2.5" rx="0.8" fill="${color}" fill-opacity="0.18"/>
            <circle cx="3"  cy="7" r="1" fill="#888" stroke="white" stroke-width="0.6"/>
            <circle cx="9" cy="7" r="1" fill="#888" stroke="white" stroke-width="0.6"/>
        </svg>`;

    return L.divIcon({
        className: isPredictable ? 'train-icon' : 'train-icon train-icon-unscheduled',
        html: svg,
        iconSize: [w, h + 2],
        iconAnchor: [w / 2, (h + 2) / 2],
    });
}

function _fmtDelay(sec) {
    if (sec === null || sec === undefined) return '—';
    const s = Math.round(sec);
    if (s < 60) return `${s} s`;
    const m = Math.floor(s / 60), r = s % 60;
    return r > 0 ? `${m}m ${r}s` : `${m} min`;
}

function _delayCls(sec) {
    if (sec === null || sec === undefined) return '';
    return sec <= 30 ? 'on-time' : sec < 120 ? 'minor-delay' : 'delayed';
}

async function openTrainPopup(train, marker) {
    const color = ROUTE_COLORS[train.route_id] || '#888888';
    const baseStopId = train.next_stop_id ? train.next_stop_id.replace(/[NS]$/, '') : null;
    const stopStation = baseStopId ? stationsById[baseStopId] : null;
    const stopName = stopStation ? stopStation.name : (train.next_stop_id || '—');
    const statusText = ({0: 'Llegando', 1: 'En estación', 2: 'En tránsito'})[train.status] ?? '—';
    const dirText = train.direction === 'N' ? '↑ Uptown' : train.direction === 'S' ? '↓ Downtown' : '';
    const schedTag = train.schedule_relationship === 1
        ? `<span class="train-sched-tag added">Servicio adicional</span>`
        : train.schedule_relationship === 2
            ? `<span class="train-sched-tag unscheduled">Sin horario</span>`
            : '';

    const popupOpts = { maxWidth: 260, className: 'train-popup-wrapper' };

    const headerHtml = `
        <div class="train-popup-header" style="background:${color}">
            <span class="train-route-badge">${train.route_id}</span>
            ${dirText ? `<span class="train-popup-dir">${dirText}</span>` : ''}
            ${schedTag}
        </div>`;

    const metaBlock = `
        <div class="train-popup-meta">
            <div class="train-popup-status">${statusText}</div>
            <div class="train-popup-stop">Próxima: <strong>${stopName}</strong></div>
        </div>`;

    // Unscheduled / added trains have no scheduled data → prediction is not possible
    if (!train.is_predictable) {
        const reason = train.schedule_relationship === 1 ? 'Servicio adicional' : 'Tren no programado';
        marker.bindPopup(
            `<div>${headerHtml}<div class="train-popup-body">${metaBlock}
                <div class="train-info-neutral">${reason} — predicción no disponible</div>
            </div></div>`,
            popupOpts
        ).openPopup();
        return;
    }

    // Show loading state immediately
    marker.bindPopup(
        `<div>${headerHtml}<div class="train-popup-body">${metaBlock}
            <div class="train-popup-loading">Cargando predicciones…</div>
        </div></div>`,
        popupOpts
    ).openPopup();

    // Call the unified per-train endpoint
    let data = null;
    try {
        const params = new URLSearchParams({ match_key: train.trip_id });
        const resp = await fetch(`/api/predict/train?${params}`);
        if (resp.ok) data = await resp.json();
    } catch (e) {
        console.error('Error fetching train predictions:', e);
    }

    // ── Build popup content from response ────────────────────────────────────
    const delayRow = (label, val) => {
        const cls  = _delayCls(val);
        const text = _fmtDelay(val);
        return `<div class="train-delay-row">
            <span class="delay-label">${label}</span>
            <span class="delay-val ${cls}">${text}</span>
        </div>`;
    };

    const fmtDelta = (d) => {
        if (!d) return `<span class="delta-chip neutral">—</span>`;
        const pct = Math.round(d.mejora_prob * 100);
        return d.mejora_predicted
            ? `<span class="delta-chip on-time">↓ ${pct}%</span>`
            : `<span class="delta-chip delayed">↑ ${100 - pct}%</span>`;
    };

    let bodyContent;
    if (!data) {
        bodyContent = `<div class="train-info-neutral">No se pudo obtener predicción</div>`;
    } else if (data.predictable === false) {
        if (data.reason_type === 'temporal') {
            bodyContent = `<div class="train-temporal">
                Datos RT momentáneamente no disponibles — espera a que el tren avance a la siguiente parada
            </div>`;
        } else {
            const msg = data.reason_type === 'ended'
                ? 'Tren finalizando recorrido'
                : 'Tren no programado';
            bodyContent = `<div class="train-info-neutral">${msg} — predicción no disponible</div>`;
        }
    } else {
        const curDelay = data.current_delay_s;
        const stopsLeft = data.stops_to_end;
        const minsLeft  = data.scheduled_time_to_end_s != null
            ? Math.round(data.scheduled_time_to_end_s / 60) : null;

        const progressHtml = (stopsLeft != null || minsLeft != null)
            ? `<div class="train-progress">${stopsLeft != null ? `${stopsLeft} paradas` : ''}${minsLeft != null ? ` · ~${minsLeft} min` : ''} hasta el final</div>`
            : '';

        const nearEnd = data.delay_30m_s == null;
        const horizonLabel = nearEnd ? 'Al llegar al final de línea' : 'En 30 min';
        const horizonDelay = nearEnd ? data.delay_end_s : data.delay_30m_s;

        const feedWarning = data.feed_warning
            ? `<div class="train-feed-warning">⚠ Datos RT no disponibles — estimación aproximada</div>`
            : '';

        bodyContent = `
            ${feedWarning}
            <div class="train-popup-meta">
                <div class="train-popup-status">${statusText}</div>
                <div class="train-popup-stop">Próxima: <strong>${stopName}</strong></div>
            </div>
            ${progressHtml}
            <div class="train-delay-grid">
                ${delayRow('Retraso actual', curDelay)}
                ${delayRow(horizonLabel, horizonDelay)}
            </div>
            <div class="train-delta-section">
                <div class="train-delta-header">Tendencia del retraso</div>
                <div class="train-delta-row"><span class="delay-label">+10 min</span>${fmtDelta(data.delta_10m)}</div>
                <div class="train-delta-row"><span class="delay-label">+20 min</span>${fmtDelta(data.delta_20m)}</div>
                <div class="train-delta-row"><span class="delay-label">+30 min</span>${fmtDelta(data.delta_30m)}</div>
            </div>`;
    }

    const fullHtml = `<div>${headerHtml}<div class="train-popup-body">${bodyContent}</div></div>`;

    const popup = marker.getPopup();
    if (popup?.isOpen()) popup.setContent(fullHtml);
}

const TRAIN_REFRESH_S = 90;
let _cdTimer = null;

function _tickCountdown(s) {
    const el = document.getElementById('train-countdown');
    if (el) el.textContent = `· ${s}s`;
    if (s > 0) _cdTimer = setTimeout(() => _tickCountdown(s - 1), 1000);
}

function _startCountdown() {
    clearTimeout(_cdTimer);
    _tickCountdown(TRAIN_REFRESH_S);
}

async function refreshTrainPositions() {
    try {
        const resp = await fetch('/api/vehicles');
        if (!resp.ok) return;
        const trains = await resp.json();

        trainLayer.clearLayers();
        trains.forEach(t => {
            if (t.lat == null || t.lon == null) return;
            const marker = L.marker([t.lat, t.lon], {
                icon: createTrainIcon(t.route_id, t.is_predictable),
                pane: 'trainPane',
            });
            const tooltipText = t.is_predictable
                ? `Línea ${t.route_id}`
                : `Línea ${t.route_id} · No programado`;
            marker.bindTooltip(tooltipText, { direction: 'top', offset: [0, -8] });
            marker.on('click', () => openTrainPopup(t, marker));
            marker.addTo(trainLayer);
        });
    } catch (e) {
        console.error('Error al obtener posiciones de trenes:', e);
    } finally {
        _startCountdown();
        setTimeout(refreshTrainPositions, TRAIN_REFRESH_S * 1000);
    }
}

refreshTrainPositions();


// ============================================================
// 9. WebSocket para predicciones en tiempo real
// ============================================================
const wsUrl = `ws://${window.location.host}/ws/live-updates`;
const socket = new WebSocket(wsUrl);

socket.onopen  = () => console.log("Conectado a FastAPI WebSocket");
socket.onclose = () => console.log("WebSocket desconectado");

socket.onmessage = (event) => {
    const payload = JSON.parse(event.data);
    if (payload.type === 'update' || payload.type === 'initial') {
        currentAlerts = payload.alerts?.predictions || [];
        updateMapWithAlerts(currentAlerts);
        if (!detailPanel.classList.contains('hidden')) {
            checkActiveAlerts();
        }
    }
};

// ============================================================
// 9. Actualización de marcadores con alertas
// ============================================================

function renderDelayCard(id, delaySeconds) {
    const el = document.getElementById(id);
    if (!el) return;
    if (delaySeconds === null || delaySeconds === undefined) {
        el.textContent = '--';
        el.className = 'delay-val';
        return;
    }
    const s = Math.round(delaySeconds);
    let text;
    if (s < 60) {
        text = `${s} s`;
    } else {
        const m = Math.floor(s / 60);
        const r = s % 60;
        text = r > 0 ? `${m} min ${r} s` : `${m} min`;
    }
    const cls = s <= 30 ? 'on-time' : (s < 120 ? 'minor-delay' : 'delayed');
    el.textContent = text;
    el.className = `delay-val ${cls}`;
}

function updateMapWithAlerts(alertPredictions) {
    // Conjunto de route_ids con alerta activa
    const alertedRoutes = new Set(
        alertPredictions.filter(a => a.alert_predicted).map(a => a.route_id)
    );

    Object.entries(markersMap).forEach(([stationId, marker]) => {
        const routes = stationRouteIndex[stationId] || [];
        const hasAlert = routes.some(r => alertedRoutes.has(r));
        const el = marker.getElement();
        if (el) el.classList.toggle('metro-delayed', hasAlert);
    });
}

// ============================================================
// 10. Panel de detalles de estación
// ============================================================

const detailPanel  = document.getElementById('station-detail-panel');
const closeDetailBtn = document.getElementById('close-detail');
closeDetailBtn.onclick = () => detailPanel.classList.add('hidden');

function openStationDetails(station) {
    detailPanel.currentStation = station;
    const nameEl        = document.getElementById('detail-station-name');
    const lineSelectors = document.getElementById('detail-line-selectors');

    detailPanel.classList.remove('hidden');
    map.flyTo([station.lat, station.lon], 14, { duration: 1.5 });

    nameEl.textContent   = station.name;
    lineSelectors.innerHTML = '';

    // Reset forecast cards immediately so old station data doesn't linger
    ['forecast-now', 'forecast-10', 'forecast-20', 'forecast-30'].forEach(id => {
        const el = document.getElementById(id);
        el.textContent = '…';
        el.className = 'delay-val';
    });
    const lbl30 = document.getElementById('forecast-30-label');
    if (lbl30) lbl30.textContent = '+30m';
    document.getElementById('station-alert').classList.add('hidden');
    document.getElementById('line-detail-info').innerHTML = '';

    const routes = station.routes.split(' ');
    routes.forEach((route, index) => {
        const bubble = document.createElement('div');
        const color  = ROUTE_COLORS[route] || '#3B82F6';
        bubble.className   = `line-bubble ${index === 0 ? 'active' : ''}`;
        bubble.textContent = route;
        bubble.style.backgroundColor = color;
        bubble.onclick = () => {
            document.querySelectorAll('.line-bubble').forEach(b => b.classList.remove('active'));
            bubble.classList.add('active');
            updateForecast(route);
        };
        lineSelectors.appendChild(bubble);
    });

    if (routes.length > 0) updateForecast(routes[0]);
}

async function updateForecast(lineCode) {
    detailPanel.currentLine = lineCode;
    const station = detailPanel.currentStation;

    ['forecast-now', 'forecast-10', 'forecast-20', 'forecast-30'].forEach(id => {
        const el = document.getElementById(id);
        el.textContent = '…';
        el.className = 'delay-val';
    });

    const loadingBar = document.getElementById('detail-loading-bar');
    if (loadingBar) loadingBar.classList.add('running');

    const stopParam = encodeURIComponent(station.id);

    // Lanzar las tres peticiones en paralelo
    const [nowResp, propResp, alertResp] = await Promise.allSettled([
        fetch(`/api/predict/current?stop_id=${stopParam}`),
        fetch(`/api/predict/propagation?stop_id=${stopParam}`),
        fetch(`/api/predict/alerts?route_id=${encodeURIComponent(lineCode)}`),
    ]);

    // Ahora: último delay observado en la ventana más reciente (en segundos)
    try {
        if (nowResp.status === 'fulfilled' && nowResp.value.ok) {
            const d = await nowResp.value.json();
            renderDelayCard('forecast-now', d.delay_seconds);
        } else {
            renderDelayCard('forecast-now', null);
        }
    } catch { renderDelayCard('forecast-now', null); }

    // +10/+20/+30 min: predicción DCRNN; fallback a LightGBM 30m si el stop no está en el grafo
    const label30 = document.getElementById('forecast-30-label');
    try {
        if (propResp.status === 'fulfilled' && propResp.value.ok) {
            const d = await propResp.value.json();
            const pred = d.predictions?.find(p => p.stop_id === station.id);
            if (pred) {
                if (label30) label30.textContent = '+30m';
                renderDelayCard('forecast-10', pred.delay_10m);
                renderDelayCard('forecast-20', pred.delay_20m);
                renderDelayCard('forecast-30', pred.delay_30m);
            } else {
                // DCRNN doesn't cover this stop — fall back to LightGBM delay 30m
                renderDelayCard('forecast-10', null);
                renderDelayCard('forecast-20', null);
                try {
                    const lgbmResp = await fetch(`/api/predict/delay/30m?stop_id=${stopParam}`);
                    if (lgbmResp.ok) {
                        const lgbm = await lgbmResp.json();
                        const p = lgbm.predictions?.[0];
                        if (label30) label30.textContent = '+30m †';
                        renderDelayCard('forecast-30', p ? p.delay_seconds : null);
                    } else {
                        if (label30) label30.textContent = '+30m';
                        renderDelayCard('forecast-30', null);
                    }
                } catch {
                    if (label30) label30.textContent = '+30m';
                    renderDelayCard('forecast-30', null);
                }
            }
        } else {
            if (label30) label30.textContent = '+30m';
            ['forecast-10', 'forecast-20', 'forecast-30'].forEach(id => renderDelayCard(id, null));
        }
    } catch {
        if (label30) label30.textContent = '+30m';
        ['forecast-10', 'forecast-20', 'forecast-30'].forEach(id => renderDelayCard(id, null));
    }

    // Probabilidad de alerta para la línea seleccionada
    try {
        if (alertResp.status === 'fulfilled' && alertResp.value.ok) {
            const d = await alertResp.value.json();
            renderAlertProbability(lineCode, d.predictions);
        } else {
            renderAlertProbability(lineCode, []);
        }
    } catch { renderAlertProbability(lineCode, []); }

    const hasDagger = label30 && label30.textContent.includes('†');
    document.getElementById('line-detail-info').innerHTML = hasDagger
        ? `<p>Línea <strong>${lineCode}</strong> · predicción DCRNN a 10, 20 min no disponible · † LightGBM 30 min</p>`
        : `<p>Línea <strong>${lineCode}</strong> · predicción DCRNN a 10, 20 y 30 min</p>`;

    if (loadingBar) loadingBar.classList.remove('running');
}

function renderAlertProbability(lineCode, predictions) {
    const section = document.getElementById('station-alert');
    const textEl  = section.querySelector('.alert-text');
    const iconEl  = section.querySelector('.alert-icon');

    if (!predictions || predictions.length === 0) {
        section.classList.add('hidden');
        return;
    }

    // Tomar la probabilidad máxima entre todas las direcciones de la línea
    const maxProb = Math.max(...predictions.map(p => p.alert_probability));
    const pct = Math.round(maxProb * 100);
    const predicted = predictions.some(p => p.alert_predicted);

    section.classList.remove('hidden');
    if (predicted) {
        iconEl.textContent = '⚠️';
        textEl.textContent = `Alerta probable en línea ${lineCode} (${pct}%)`;
        section.querySelector('.alert-banner').style.background = 'rgba(239,68,68,0.15)';
    } else {
        iconEl.textContent = '✓';
        textEl.textContent = `Sin alerta en línea ${lineCode} (${pct}% prob.)`;
        section.querySelector('.alert-banner').style.background = 'rgba(34,197,94,0.1)';
    }
}

function checkActiveAlerts() {
    const alertSection = document.getElementById('station-alert');
    if (!detailPanel.currentStation || !detailPanel.currentLine) return;
    const stationRoutes = stationRouteIndex[detailPanel.currentStation.id] || [];
    const hasAlert = currentAlerts.some(a => a.alert_predicted && stationRoutes.includes(a.route_id));
    alertSection.classList.toggle('hidden', !hasAlert);
}

// ============================================================
// 11. Planificador de rutas con autocompletado
// ============================================================

function initRoutePlanner() {
    setupAutocomplete(
        document.getElementById('origin-input'),
        document.getElementById('origin-results')
    );
    setupAutocomplete(
        document.getElementById('destination-input'),
        document.getElementById('destination-results')
    );
}

function setupAutocomplete(input, dropdown) {
    input.addEventListener('input', () => {
        const value = input.value.toLowerCase().trim();
        if (value.length < 2) { dropdown.classList.add('hidden'); return; }

        const filtered = allStations.filter(s => s.name.toLowerCase().includes(value)).slice(0, 5);

        if (filtered.length > 0) {
            dropdown.innerHTML = '';
            filtered.forEach(s => {
                const item = document.createElement('div');
                item.className   = 'suggestion-item';
                item.textContent = s.name;
                item.onclick = () => {
                    input.value = s.name;
                    input.dataset.stationId = s.id;
                    dropdown.classList.add('hidden');
                    calculateRoute();
                };
                dropdown.appendChild(item);
            });
            dropdown.classList.remove('hidden');
        } else {
            dropdown.classList.add('hidden');
        }
    });

    document.addEventListener('click', (e) => {
        if (!input.contains(e.target) && !dropdown.contains(e.target))
            dropdown.classList.add('hidden');
    });
}

async function calculateRoute() {
    const originId  = document.getElementById('origin-input').dataset.stationId;
    const destId    = document.getElementById('destination-input').dataset.stationId;
    const resultPanel = document.getElementById('route-prediction');

    if (!originId || !destId) return;

    const origin      = allStations.find(s => s.id === originId);
    const destination = allStations.find(s => s.id === destId);

    resultPanel.classList.remove('hidden');

    const commonLines = findCommonLines(origin, destination);
    const displayLine = commonLines.length > 0 ? commonLines[0] : origin.routes.split(' ')[0];

    document.getElementById('result-line').textContent = displayLine;
    document.getElementById('result-line').style.backgroundColor = ROUTE_COLORS[displayLine] || '#3B82F6';
    document.getElementById('result-station').textContent = origin.name;

    renderDelayCard('result-delay-val', null);

    try {
        const resp = await fetch(`/api/predict/delay/30m?stop_id=${encodeURIComponent(originId)}`);
        if (!resp.ok) throw new Error(resp.statusText);
        const data = await resp.json();
        const pred = data.predictions?.[0];
        renderDelayCard('result-delay-val', pred ? pred.delay_minutes * 60 : null);
    } catch (e) {
        console.error('Error al obtener retraso de ruta:', e);
    }
}

function findCommonLines(s1, s2) {
    const r1 = s1.routes.split(' ');
    const r2 = s2.routes.split(' ');
    return r1.filter(line => r2.includes(line));
}

// ============================================================
// 12. Modo claro / oscuro
// ============================================================

const ICON_MOON = `<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>`;
const ICON_SUN  = `<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>`;

function applyTheme(dark) {
    const btn = document.getElementById('theme-toggle');
    if (dark) {
        document.documentElement.setAttribute('data-theme', 'dark');
        tileLayer.setUrl(TILE_DARK);
        btn.innerHTML = ICON_SUN;
        btn.title = 'Modo claro';
    } else {
        document.documentElement.removeAttribute('data-theme');
        tileLayer.setUrl(TILE_LIGHT);
        btn.innerHTML = ICON_MOON;
        btn.title = 'Modo oscuro';
    }
}

(function initTheme() {
    const saved = localStorage.getItem('theme');
    applyTheme(saved === 'dark');
})();

document.getElementById('theme-toggle').addEventListener('click', () => {
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    const next = !isDark;
    localStorage.setItem('theme', next ? 'dark' : 'light');
    applyTheme(next);
});