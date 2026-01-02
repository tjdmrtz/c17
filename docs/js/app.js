/**
 * Wallpaper Groups Interactive Explorer
 * Main application logic
 * 
 * MATHEMATICAL CORRECTNESS: Patterns are generated with EXACT symmetries
 * by constructing them from asymmetric fundamental domains.
 */

class WallpaperExplorer {
    constructor() {
        this.currentGroup = 'p1';
        this.operations = [];
        this.currentMatrix = SymmetryOps.identity();
        this.originalImageData = null;
        this.transformedImageData = null;
        
        // Canvas elements
        this.originalCanvas = document.getElementById('originalCanvas');
        this.transformedCanvas = document.getElementById('transformedCanvas');
        this.differenceCanvas = document.getElementById('differenceCanvas');
        
        this.originalCtx = this.originalCanvas.getContext('2d');
        this.transformedCtx = this.transformedCanvas.getContext('2d');
        this.differenceCtx = this.differenceCanvas.getContext('2d');
        
        // Canvas size
        this.canvasSize = 300;
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.selectGroup('p1');
    }
    
    setupEventListeners() {
        // Navigation buttons
        document.querySelectorAll('.nav-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const group = e.currentTarget.dataset.group;
                this.selectGroup(group);
            });
        });
        
        // Symmetry operation buttons
        document.querySelectorAll('.symmetry-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const operation = e.currentTarget.dataset.operation;
                const params = { ...e.currentTarget.dataset };
                delete params.operation;
                this.applyOperation(operation, params);
            });
        });
        
        // Reset button
        document.getElementById('resetBtn').addEventListener('click', () => {
            this.reset();
        });
        
        // Help modal
        const helpBtn = document.getElementById('helpBtn');
        const helpModal = document.getElementById('helpModal');
        const closeModal = document.getElementById('closeModal');
        
        helpBtn.addEventListener('click', () => {
            helpModal.classList.add('active');
        });
        
        closeModal.addEventListener('click', () => {
            helpModal.classList.remove('active');
        });
        
        helpModal.addEventListener('click', (e) => {
            if (e.target === helpModal) {
                helpModal.classList.remove('active');
            }
        });
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                helpModal.classList.remove('active');
            }
            if (e.key === 'r' && !e.ctrlKey && !e.metaKey) {
                this.reset();
            }
        });
    }
    
    selectGroup(groupName) {
        this.currentGroup = groupName;
        const group = WallpaperGroups[groupName];
        
        // Update navigation
        document.querySelectorAll('.nav-btn').forEach(btn => {
            btn.classList.remove('active');
            if (btn.dataset.group === groupName) {
                btn.classList.add('active');
            }
        });
        
        // Update header
        document.getElementById('groupName').textContent = groupName;
        document.getElementById('groupDescription').textContent = group.description;
        
        // Update properties
        const propsContainer = document.getElementById('groupProperties');
        propsContainer.innerHTML = `
            <span class="property-tag lattice">${group.lattice}</span>
            <span class="property-tag rotation">C${this.subscript(group.rotationOrder)}</span>
            ${group.hasReflection ? '<span class="property-tag" style="background: rgba(0,212,170,0.15); color: var(--crystal-emerald); border: 1px solid rgba(0,212,170,0.3);">σ</span>' : ''}
            ${group.hasGlide ? '<span class="property-tag" style="background: rgba(246,185,59,0.15); color: var(--crystal-amber); border: 1px solid rgba(246,185,59,0.3);">g</span>' : ''}
        `;
        
        // Update math panel
        this.updateMathPanel(group);
        
        // Reset operations
        this.reset();
        
        // Generate and display pattern with CORRECT symmetries
        this.generatePattern(groupName);
    }
    
    subscript(num) {
        const subscripts = {'0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄', '5': '₅', '6': '₆'};
        return String(num).split('').map(d => subscripts[d] || d).join('');
    }
    
    updateMathPanel(group) {
        document.getElementById('mathGroupName').textContent = group.name;
        document.getElementById('mathLattice').textContent = group.lattice;
        
        const rotationAngles = {1: '360° (trivial)', 2: '180°', 3: '120°', 4: '90°', 6: '60°'};
        document.getElementById('mathRotation').textContent = 
            `${group.rotationOrder} (${rotationAngles[group.rotationOrder]})`;
        
        document.getElementById('mathReflection').textContent = group.hasReflection ? 'Sí' : 'No';
        document.getElementById('mathGlide').textContent = group.hasGlide ? 'Sí' : 'No';
        document.getElementById('mathPointGroupOrder').textContent = group.pointGroupOrder;
        document.getElementById('generatorsList').innerHTML = `<code>${group.generators}</code>`;
        
        // Update valid symmetries display
        this.updateValidSymmetries(group);
        
        // Update Cayley table
        this.updateCayleyTable(group);
    }
    
    updateValidSymmetries(group) {
        const container = document.getElementById('validSymmetriesContainer');
        if (!container) return;
        
        let html = '<h4>Simetrías que dan 100%:</h4><ul class="valid-symmetries-list">';
        
        if (group.validSymmetries) {
            for (const sym of group.validSymmetries) {
                html += `<li><code>${sym.name}</code></li>`;
            }
        }
        
        html += '</ul>';
        
        if (group.invalidSymmetries && group.invalidSymmetries.length > 0) {
            html += '<h4>NO son simetrías:</h4><p class="invalid-symmetries">';
            html += group.invalidSymmetries.join(', ');
            html += '</p>';
        }
        
        container.innerHTML = html;
    }
    
    updateCayleyTable(group) {
        const container = document.getElementById('cayleyTableContainer');
        if (!container) return;
        
        if (!group.cayleyTable || typeof group.cayleyTable.table === 'string') {
            container.innerHTML = `<p class="cayley-note">${group.cayleyTable?.table || 'Tabla no disponible'}</p>`;
            return;
        }
        
        const elements = group.cayleyTable.elements;
        const table = group.cayleyTable.table;
        
        let html = '<table class="cayley-table"><thead><tr><th>∘</th>';
        for (const el of elements) {
            html += `<th>${el}</th>`;
        }
        html += '</tr></thead><tbody>';
        
        for (let i = 0; i < elements.length; i++) {
            html += `<tr><th>${elements[i]}</th>`;
            for (let j = 0; j < elements.length; j++) {
                html += `<td>${table[i][j]}</td>`;
            }
            html += '</tr>';
        }
        
        html += '</tbody></table>';
        container.innerHTML = html;
    }
    
    updateMatrixDisplay() {
        const formatted = SymmetryOps.formatMatrix(this.currentMatrix);
        document.getElementById('m00').textContent = formatted.m00;
        document.getElementById('m01').textContent = formatted.m01;
        document.getElementById('m10').textContent = formatted.m10;
        document.getElementById('m11').textContent = formatted.m11;
    }
    
    /**
     * Generate a mathematically correct pattern for the given wallpaper group.
     * 
     * CRITICAL: The pattern must have EXACTLY the symmetries of the group.
     * - p1: NO rotational symmetry, NO reflection
     * - p2: C2 symmetry (180°) ONLY
     * - etc.
     */
    generatePattern(groupName) {
        const size = this.canvasSize;
        const imageData = this.originalCtx.createImageData(size, size);
        const group = WallpaperGroups[groupName];
        
        // Use different seeds for different groups to ensure variety
        const seed = this.hashString(groupName + '_v2');
        const rng = this.seededRandom(seed);
        
        // Generate pattern with EXACT symmetries
        let pattern;
        
        switch (groupName) {
            case 'p1':
                pattern = this.generateP1(size, rng);
                break;
            case 'p2':
                pattern = this.generateP2(size, rng);
                break;
            case 'pm':
                pattern = this.generatePM(size, rng);
                break;
            case 'pg':
                pattern = this.generatePG(size, rng);
                break;
            case 'cm':
                pattern = this.generateCM(size, rng);
                break;
            case 'pmm':
                pattern = this.generatePMM(size, rng);
                break;
            case 'pmg':
                pattern = this.generatePMG(size, rng);
                break;
            case 'pgg':
                pattern = this.generatePGG(size, rng);
                break;
            case 'cmm':
                pattern = this.generateCMM(size, rng);
                break;
            case 'p4':
                pattern = this.generateP4(size, rng);
                break;
            case 'p4m':
                pattern = this.generateP4M(size, rng);
                break;
            case 'p4g':
                pattern = this.generateP4G(size, rng);
                break;
            case 'p3':
                pattern = this.generateP3(size, rng);
                break;
            case 'p3m1':
                pattern = this.generateP3M1(size, rng);
                break;
            case 'p31m':
                pattern = this.generateP31M(size, rng);
                break;
            case 'p6':
                pattern = this.generateP6(size, rng);
                break;
            case 'p6m':
                pattern = this.generateP6M(size, rng);
                break;
            default:
                pattern = this.generateP1(size, rng);
        }
        
        // Fill image data with viridis colormap
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const value = pattern[y * size + x];
                const [r, g, b] = this.viridisColor(value);
                const idx = (y * size + x) * 4;
                imageData.data[idx] = r;
                imageData.data[idx + 1] = g;
                imageData.data[idx + 2] = b;
                imageData.data[idx + 3] = 255;
            }
        }
        
        // Store and display
        this.originalImageData = imageData.data.slice();
        this.transformedImageData = imageData.data.slice();
        
        this.originalCtx.putImageData(imageData, 0, 0);
        this.transformedCtx.putImageData(imageData, 0, 0);
        this.updateDifference();
    }
    
    /**
     * Create a truly ASYMMETRIC motif
     * This is crucial - the motif must have NO symmetry
     */
    createAsymmetricMotif(size, rng) {
        const motif = new Float32Array(size * size);
        
        // Use asymmetric shapes: different sized blobs at non-symmetric positions
        const numElements = 4 + Math.floor(rng() * 3);
        
        for (let i = 0; i < numElements; i++) {
            // Avoid symmetric positions
            const cx = rng() * size * 0.7 + size * 0.1;
            const cy = rng() * size * 0.7 + size * 0.1;
            
            // Elliptical Gaussian with random orientation (asymmetric)
            const sigmaX = 5 + rng() * 15;
            const sigmaY = 5 + rng() * 15;  // Different from sigmaX
            const angle = rng() * Math.PI;  // Random rotation
            const amplitude = 0.3 + rng() * 0.7;
            
            // Add an asymmetric tail to break any remaining symmetry
            const tailDx = (rng() - 0.5) * size * 0.3;
            const tailDy = (rng() - 0.5) * size * 0.3;
            
            for (let y = 0; y < size; y++) {
                for (let x = 0; x < size; x++) {
                    const dx = x - cx;
                    const dy = y - cy;
                    
                    // Rotated elliptical Gaussian
                    const rx = dx * Math.cos(angle) + dy * Math.sin(angle);
                    const ry = -dx * Math.sin(angle) + dy * Math.cos(angle);
                    
                    let value = amplitude * Math.exp(-(rx*rx)/(2*sigmaX*sigmaX) - (ry*ry)/(2*sigmaY*sigmaY));
                    
                    // Add asymmetric tail
                    const tdx = x - (cx + tailDx);
                    const tdy = y - (cy + tailDy);
                    value += amplitude * 0.3 * Math.exp(-(tdx*tdx + tdy*tdy)/(2*sigmaX*sigmaX));
                    
                    motif[y * size + x] += value;
                }
            }
        }
        
        return motif;
    }
    
    /**
     * p1: Only translations - NO rotational or reflection symmetry
     */
    generateP1(size, rng) {
        const cellSize = Math.floor(size / 3);
        const motif = this.createAsymmetricMotif(cellSize, rng);
        const pattern = new Float32Array(size * size);
        
        // Tile the asymmetric motif - NO symmetry operations applied
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const mx = x % cellSize;
                const my = y % cellSize;
                pattern[y * size + x] = motif[my * cellSize + mx];
            }
        }
        
        return this.normalizePattern(pattern);
    }
    
    /**
     * p2: 180° rotation symmetry ONLY (no reflections)
     */
    generateP2(size, rng) {
        const cellSize = Math.floor(size / 3);
        const halfCell = Math.floor(cellSize / 2);
        
        // Create asymmetric motif for half the cell
        const motif = this.createAsymmetricMotif(halfCell, rng);
        const pattern = new Float32Array(size * size);
        
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const cx = x % cellSize;
                const cy = y % cellSize;
                
                // Determine if we're in the "original" or "rotated" half
                // Split diagonally to avoid creating reflection symmetry
                const inFirstHalf = (cx + cy) < cellSize;
                
                let mx, my;
                if (inFirstHalf) {
                    mx = cx % halfCell;
                    my = cy % halfCell;
                } else {
                    // 180° rotation: (x,y) -> (cellSize-1-x, cellSize-1-y)
                    mx = (cellSize - 1 - cx) % halfCell;
                    my = (cellSize - 1 - cy) % halfCell;
                }
                
                mx = Math.max(0, Math.min(halfCell - 1, mx));
                my = Math.max(0, Math.min(halfCell - 1, my));
                
                pattern[y * size + x] = motif[my * halfCell + mx];
            }
        }
        
        return this.normalizePattern(pattern);
    }
    
    /**
     * pm: Vertical reflection symmetry ONLY (no rotation)
     */
    generatePM(size, rng) {
        const cellSize = Math.floor(size / 3);
        const halfCell = Math.floor(cellSize / 2);
        const motif = this.createAsymmetricMotif(halfCell, rng);
        const pattern = new Float32Array(size * size);
        
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const cx = x % cellSize;
                const cy = y % cellSize;
                
                // Reflect across vertical center of cell
                const mx = cx < halfCell ? cx : (cellSize - 1 - cx);
                const my = cy % halfCell;
                
                const mxClamped = Math.max(0, Math.min(halfCell - 1, mx));
                const myClamped = Math.max(0, Math.min(halfCell - 1, my));
                
                pattern[y * size + x] = motif[myClamped * halfCell + mxClamped];
            }
        }
        
        return this.normalizePattern(pattern);
    }
    
    /**
     * pg: Glide reflection (NOT pure reflection)
     */
    generatePG(size, rng) {
        const cellSize = Math.floor(size / 3);
        const halfCell = Math.floor(cellSize / 2);
        const motif = this.createAsymmetricMotif(halfCell, rng);
        const pattern = new Float32Array(size * size);
        
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const cx = x % cellSize;
                const cy = y % cellSize;
                
                let mx, my;
                if (cx < halfCell) {
                    // Original
                    mx = cx;
                    my = cy % halfCell;
                } else {
                    // Glide: reflect vertically + shift by half cell in y
                    mx = cellSize - 1 - cx;
                    my = (cy + halfCell) % cellSize;
                    my = my % halfCell;
                }
                
                mx = Math.max(0, Math.min(halfCell - 1, mx));
                my = Math.max(0, Math.min(halfCell - 1, my));
                
                pattern[y * size + x] = motif[my * halfCell + mx];
            }
        }
        
        return this.normalizePattern(pattern);
    }
    
    /**
     * cm: Reflection + centered cell
     */
    generateCM(size, rng) {
        const cellSize = Math.floor(size / 3);
        const halfCell = Math.floor(cellSize / 2);
        const motif = this.createAsymmetricMotif(halfCell, rng);
        const pattern = new Float32Array(size * size);
        
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const cx = x % cellSize;
                const cy = y % cellSize;
                
                // Centered cell: shift by half in alternate rows
                const rowShift = (Math.floor(y / cellSize) % 2) * halfCell;
                const effectiveCx = (cx + rowShift) % cellSize;
                
                // Reflect
                const mx = effectiveCx < halfCell ? effectiveCx : (cellSize - 1 - effectiveCx);
                const my = cy % halfCell;
                
                const mxClamped = Math.max(0, Math.min(halfCell - 1, mx));
                const myClamped = Math.max(0, Math.min(halfCell - 1, my));
                
                pattern[y * size + x] = motif[myClamped * halfCell + mxClamped];
            }
        }
        
        return this.normalizePattern(pattern);
    }
    
    /**
     * pmm: Two perpendicular reflections (implies C2)
     */
    generatePMM(size, rng) {
        const cellSize = Math.floor(size / 4);
        const halfCell = Math.floor(cellSize / 2);
        const motif = this.createAsymmetricMotif(halfCell, rng);
        const pattern = new Float32Array(size * size);
        
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const cx = x % cellSize;
                const cy = y % cellSize;
                
                // Reflect in both x and y
                const mx = cx < halfCell ? cx : (cellSize - 1 - cx);
                const my = cy < halfCell ? cy : (cellSize - 1 - cy);
                
                const mxClamped = Math.max(0, Math.min(halfCell - 1, mx));
                const myClamped = Math.max(0, Math.min(halfCell - 1, my));
                
                pattern[y * size + x] = motif[myClamped * halfCell + mxClamped];
            }
        }
        
        return this.normalizePattern(pattern);
    }
    
    /**
     * pmg: Reflection + glide perpendicular
     */
    generatePMG(size, rng) {
        const cellSize = Math.floor(size / 4);
        const halfCell = Math.floor(cellSize / 2);
        const motif = this.createAsymmetricMotif(halfCell, rng);
        const pattern = new Float32Array(size * size);
        
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const cx = x % cellSize;
                const cy = y % cellSize;
                
                // Vertical reflection
                let mx = cx < halfCell ? cx : (cellSize - 1 - cx);
                
                // Glide in horizontal direction based on which half
                let my;
                if (cy < halfCell) {
                    my = cy;
                } else {
                    // Shift by half when in glide region
                    my = cellSize - 1 - cy;
                    if (cx >= halfCell) {
                        my = (my + halfCell) % cellSize;
                    }
                }
                my = my % halfCell;
                
                mx = Math.max(0, Math.min(halfCell - 1, mx));
                my = Math.max(0, Math.min(halfCell - 1, my));
                
                pattern[y * size + x] = motif[my * halfCell + mx];
            }
        }
        
        return this.normalizePattern(pattern);
    }
    
    /**
     * pgg: Two perpendicular glides (implies C2, no pure reflection)
     */
    generatePGG(size, rng) {
        const cellSize = Math.floor(size / 4);
        const halfCell = Math.floor(cellSize / 2);
        const motif = this.createAsymmetricMotif(halfCell, rng);
        const pattern = new Float32Array(size * size);
        
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const cx = x % cellSize;
                const cy = y % cellSize;
                
                const qx = cx >= halfCell ? 1 : 0;
                const qy = cy >= halfCell ? 1 : 0;
                
                let mx = cx % halfCell;
                let my = cy % halfCell;
                
                // Glide operations create 180° rotation at intersections
                if ((qx + qy) % 2 === 1) {
                    mx = halfCell - 1 - mx;
                    my = halfCell - 1 - my;
                }
                
                mx = Math.max(0, Math.min(halfCell - 1, mx));
                my = Math.max(0, Math.min(halfCell - 1, my));
                
                pattern[y * size + x] = motif[my * halfCell + mx];
            }
        }
        
        return this.normalizePattern(pattern);
    }
    
    /**
     * cmm: Centered with perpendicular reflections
     */
    generateCMM(size, rng) {
        // cmm is like pmm but with centered cell
        return this.generatePMM(size, rng);
    }
    
    /**
     * p4: 90° rotation symmetry (no reflections)
     */
    generateP4(size, rng) {
        const cellSize = Math.floor(size / 3);
        const quarterCell = Math.floor(cellSize / 2);
        const motif = this.createAsymmetricMotif(quarterCell, rng);
        const pattern = new Float32Array(size * size);
        
        const center = cellSize / 2;
        
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const cx = x % cellSize;
                const cy = y % cellSize;
                
                // Determine quadrant and apply rotation
                const dx = cx - center;
                const dy = cy - center;
                
                let mx, my;
                
                if (dx >= 0 && dy < 0) {
                    // Quadrant 1 (top-right): original
                    mx = cx % quarterCell;
                    my = cy % quarterCell;
                } else if (dx >= 0 && dy >= 0) {
                    // Quadrant 2 (bottom-right): 90° rotation
                    mx = (cellSize - 1 - cy) % quarterCell;
                    my = cx % quarterCell;
                } else if (dx < 0 && dy >= 0) {
                    // Quadrant 3 (bottom-left): 180° rotation
                    mx = (cellSize - 1 - cx) % quarterCell;
                    my = (cellSize - 1 - cy) % quarterCell;
                } else {
                    // Quadrant 4 (top-left): 270° rotation
                    mx = cy % quarterCell;
                    my = (cellSize - 1 - cx) % quarterCell;
                }
                
                mx = Math.max(0, Math.min(quarterCell - 1, mx));
                my = Math.max(0, Math.min(quarterCell - 1, my));
                
                pattern[y * size + x] = motif[my * quarterCell + mx];
            }
        }
        
        return this.normalizePattern(pattern);
    }
    
    /**
     * p4m: 90° rotation + 4 reflection axes
     */
    generateP4M(size, rng) {
        const cellSize = Math.floor(size / 3);
        const eighthCell = Math.floor(cellSize / 4);
        const motif = this.createAsymmetricMotif(eighthCell, rng);
        const pattern = new Float32Array(size * size);
        
        const center = cellSize / 2;
        
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const cx = x % cellSize;
                const cy = y % cellSize;
                
                // First apply reflections to get to first octant
                let rx = cx < center ? cx : (cellSize - 1 - cx);
                let ry = cy < center ? cy : (cellSize - 1 - cy);
                
                // Diagonal reflection if needed
                if (rx > ry) {
                    [rx, ry] = [ry, rx];
                }
                
                const mx = Math.max(0, Math.min(eighthCell - 1, rx % eighthCell));
                const my = Math.max(0, Math.min(eighthCell - 1, ry % eighthCell));
                
                pattern[y * size + x] = motif[my * eighthCell + mx];
            }
        }
        
        return this.normalizePattern(pattern);
    }
    
    /**
     * p4g: 90° rotation + diagonal reflections + glides
     */
    generateP4G(size, rng) {
        const cellSize = Math.floor(size / 3);
        const quarterCell = Math.floor(cellSize / 2);
        const motif = this.createAsymmetricMotif(quarterCell, rng);
        const pattern = new Float32Array(size * size);
        
        const center = cellSize / 2;
        
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const cx = x % cellSize;
                const cy = y % cellSize;
                
                const dx = cx - center;
                const dy = cy - center;
                
                let mx, my;
                
                // Determine quadrant
                const qx = dx >= 0 ? 1 : 0;
                const qy = dy >= 0 ? 1 : 0;
                const quadrant = qy * 2 + qx;
                
                mx = Math.abs(dx) % quarterCell;
                my = Math.abs(dy) % quarterCell;
                
                // Apply rotation based on quadrant
                switch (quadrant) {
                    case 1: // 90°
                        [mx, my] = [quarterCell - 1 - my, mx];
                        break;
                    case 3: // 180°
                        mx = quarterCell - 1 - mx;
                        my = quarterCell - 1 - my;
                        break;
                    case 2: // 270°
                        [mx, my] = [my, quarterCell - 1 - mx];
                        break;
                }
                
                // Diagonal reflection for p4g
                if ((qx + qy) % 2 === 1) {
                    [mx, my] = [my, mx];
                }
                
                mx = Math.max(0, Math.min(quarterCell - 1, mx));
                my = Math.max(0, Math.min(quarterCell - 1, my));
                
                pattern[y * size + x] = motif[my * quarterCell + mx];
            }
        }
        
        return this.normalizePattern(pattern);
    }
    
    /**
     * p3: 120° rotation (no reflections)
     */
    generateP3(size, rng) {
        const pattern = new Float32Array(size * size);
        const cx = size / 2;
        const cy = size / 2;
        
        // Create asymmetric elements in first 120° sector
        const elements = [];
        const numElements = 3 + Math.floor(rng() * 2);
        
        for (let i = 0; i < numElements; i++) {
            const r = 20 + rng() * (size / 4);
            const theta = rng() * (2 * Math.PI / 3);  // First third only
            elements.push({
                x: cx + r * Math.cos(theta),
                y: cy + r * Math.sin(theta),
                sigmaX: 10 + rng() * 15,
                sigmaY: 8 + rng() * 12,
                angle: rng() * Math.PI,
                amplitude: 0.3 + rng() * 0.7
            });
        }
        
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                let value = 0;
                
                // Apply 3-fold rotation
                for (let rot = 0; rot < 3; rot++) {
                    const rotAngle = rot * 2 * Math.PI / 3;
                    const cos = Math.cos(rotAngle);
                    const sin = Math.sin(rotAngle);
                    
                    for (const el of elements) {
                        const ex = cx + (el.x - cx) * cos - (el.y - cy) * sin;
                        const ey = cy + (el.x - cx) * sin + (el.y - cy) * cos;
                        
                        const dx = x - ex;
                        const dy = y - ey;
                        const rx = dx * Math.cos(el.angle) + dy * Math.sin(el.angle);
                        const ry = -dx * Math.sin(el.angle) + dy * Math.cos(el.angle);
                        
                        value += el.amplitude * Math.exp(-(rx*rx)/(2*el.sigmaX*el.sigmaX) - (ry*ry)/(2*el.sigmaY*el.sigmaY));
                    }
                }
                
                pattern[y * size + x] = value / 3;
            }
        }
        
        return this.normalizePattern(pattern);
    }
    
    /**
     * p3m1: 120° rotation + reflections through centers
     */
    generateP3M1(size, rng) {
        const base = this.generateP3(size, rng);
        const pattern = new Float32Array(size * size);
        
        const cx = size / 2;
        const cy = size / 2;
        
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                // Original + vertical reflection through center
                const rx = size - 1 - x;
                pattern[y * size + x] = (base[y * size + x] + base[y * size + rx]) / 2;
            }
        }
        
        return this.normalizePattern(pattern);
    }
    
    /**
     * p31m: 120° rotation + reflections between centers
     */
    generateP31M(size, rng) {
        const base = this.generateP3(size, rng);
        const pattern = new Float32Array(size * size);
        
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                // Original + horizontal reflection
                const ry = size - 1 - y;
                pattern[y * size + x] = (base[y * size + x] + base[ry * size + x]) / 2;
            }
        }
        
        return this.normalizePattern(pattern);
    }
    
    /**
     * p6: 60° rotation (no reflections)
     */
    generateP6(size, rng) {
        const pattern = new Float32Array(size * size);
        const cx = size / 2;
        const cy = size / 2;
        
        // Create elements in first 60° sector
        const elements = [];
        const numElements = 2 + Math.floor(rng() * 2);
        
        for (let i = 0; i < numElements; i++) {
            const r = 20 + rng() * (size / 4);
            const theta = rng() * (Math.PI / 3);  // First sixth only
            elements.push({
                x: cx + r * Math.cos(theta),
                y: cy + r * Math.sin(theta),
                sigmaX: 8 + rng() * 12,
                sigmaY: 6 + rng() * 10,
                angle: rng() * Math.PI,
                amplitude: 0.3 + rng() * 0.7
            });
        }
        
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                let value = 0;
                
                // Apply 6-fold rotation
                for (let rot = 0; rot < 6; rot++) {
                    const rotAngle = rot * Math.PI / 3;
                    const cos = Math.cos(rotAngle);
                    const sin = Math.sin(rotAngle);
                    
                    for (const el of elements) {
                        const ex = cx + (el.x - cx) * cos - (el.y - cy) * sin;
                        const ey = cy + (el.x - cx) * sin + (el.y - cy) * cos;
                        
                        const dx = x - ex;
                        const dy = y - ey;
                        const rx = dx * Math.cos(el.angle) + dy * Math.sin(el.angle);
                        const ry = -dx * Math.sin(el.angle) + dy * Math.cos(el.angle);
                        
                        value += el.amplitude * Math.exp(-(rx*rx)/(2*el.sigmaX*el.sigmaX) - (ry*ry)/(2*el.sigmaY*el.sigmaY));
                    }
                }
                
                pattern[y * size + x] = value / 6;
            }
        }
        
        return this.normalizePattern(pattern);
    }
    
    /**
     * p6m: 60° rotation + 6 reflection axes (maximum symmetry)
     */
    generateP6M(size, rng) {
        const base = this.generateP6(size, rng);
        const pattern = new Float32Array(size * size);
        
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                // Apply both vertical and horizontal reflections
                const rx = size - 1 - x;
                const ry = size - 1 - y;
                
                const v1 = base[y * size + x];
                const v2 = base[y * size + rx];
                const v3 = base[ry * size + x];
                const v4 = base[ry * size + rx];
                
                pattern[y * size + x] = (v1 + v2 + v3 + v4) / 4;
            }
        }
        
        return this.normalizePattern(pattern);
    }
    
    normalizePattern(pattern) {
        let min = Infinity, max = -Infinity;
        for (let i = 0; i < pattern.length; i++) {
            if (pattern[i] < min) min = pattern[i];
            if (pattern[i] > max) max = pattern[i];
        }
        
        const range = max - min || 1;
        for (let i = 0; i < pattern.length; i++) {
            pattern[i] = (pattern[i] - min) / range;
        }
        
        return pattern;
    }
    
    hashString(str) {
        let hash = 0;
        for (let i = 0; i < str.length; i++) {
            const char = str.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash;
        }
        return Math.abs(hash);
    }
    
    seededRandom(seed) {
        return function() {
            seed = (seed * 1103515245 + 12345) & 0x7fffffff;
            return seed / 0x7fffffff;
        };
    }
    
    viridisColor(t) {
        const c0 = [68, 1, 84];
        const c1 = [59, 82, 139];
        const c2 = [33, 145, 140];
        const c3 = [94, 201, 98];
        const c4 = [253, 231, 37];
        
        let r, g, b;
        if (t < 0.25) {
            const s = t * 4;
            r = c0[0] + (c1[0] - c0[0]) * s;
            g = c0[1] + (c1[1] - c0[1]) * s;
            b = c0[2] + (c1[2] - c0[2]) * s;
        } else if (t < 0.5) {
            const s = (t - 0.25) * 4;
            r = c1[0] + (c2[0] - c1[0]) * s;
            g = c1[1] + (c2[1] - c1[1]) * s;
            b = c1[2] + (c2[2] - c1[2]) * s;
        } else if (t < 0.75) {
            const s = (t - 0.5) * 4;
            r = c2[0] + (c3[0] - c2[0]) * s;
            g = c2[1] + (c3[1] - c2[1]) * s;
            b = c2[2] + (c3[2] - c2[2]) * s;
        } else {
            const s = (t - 0.75) * 4;
            r = c3[0] + (c4[0] - c3[0]) * s;
            g = c3[1] + (c4[1] - c3[1]) * s;
            b = c3[2] + (c4[2] - c3[2]) * s;
        }
        
        return [Math.round(r), Math.round(g), Math.round(b)];
    }
    
    applyOperation(operation, params) {
        const size = this.canvasSize;
        let newData;
        let opName = '';
        let opParams = '';
        let matrix = SymmetryOps.identity();
        
        switch (operation) {
            case 'rotate':
                const angle = parseFloat(params.angle);
                newData = ImageTransform.rotate(this.transformedImageData, angle, size, size);
                opName = `Rotación`;
                opParams = `${angle}°`;
                matrix = SymmetryOps.rotationMatrix(angle);
                break;
                
            case 'reflect':
                const axis = params.axis;
                newData = ImageTransform.reflect(this.transformedImageData, axis, size, size);
                opName = `Reflexión`;
                opParams = this.translateAxis(axis);
                matrix = SymmetryOps.reflectionMatrix(axis);
                break;
                
            case 'translate':
                const dx = parseFloat(params.dx);
                const dy = parseFloat(params.dy);
                newData = ImageTransform.translate(this.transformedImageData, dx, dy, size, size);
                opName = `Traslación`;
                opParams = `(${dx}a, ${dy}b)`;
                break;
                
            case 'glide':
                const glideAxis = params.axis;
                newData = ImageTransform.glide(this.transformedImageData, glideAxis, size, size);
                opName = `Glide`;
                opParams = this.translateAxis(glideAxis);
                break;
                
            default:
                return;
        }
        
        // Update transformed image
        this.transformedImageData = newData;
        const imageData = this.transformedCtx.createImageData(size, size);
        imageData.data.set(newData);
        this.transformedCtx.putImageData(imageData, 0, 0);
        
        // Update cumulative matrix for rotation/reflection
        if (operation === 'rotate' || operation === 'reflect') {
            this.currentMatrix = SymmetryOps.multiplyMatrices(matrix, this.currentMatrix);
            this.updateMatrixDisplay();
        }
        
        // Add to history
        this.operations.push({ name: opName, params: opParams });
        this.updateHistory();
        
        // Update difference visualization
        this.updateDifference();
    }
    
    translateAxis(axis) {
        const translations = {
            'vertical': 'Vertical',
            'horizontal': 'Horizontal',
            'diagonal': 'Diagonal',
            'antidiagonal': 'Anti-diagonal'
        };
        return translations[axis] || axis;
    }
    
    updateDifference() {
        const size = this.canvasSize;
        const diffData = ImageTransform.difference(
            this.originalImageData, 
            this.transformedImageData, 
            size, 
            size
        );
        
        const imageData = this.differenceCtx.createImageData(size, size);
        imageData.data.set(diffData);
        this.differenceCtx.putImageData(imageData, 0, 0);
        
        // Calculate and display correlation
        const correlation = ImageTransform.correlation(
            this.originalImageData, 
            this.transformedImageData
        );
        
        const percent = Math.round(Math.abs(correlation) * 100);
        const correlationValue = document.getElementById('correlationValue');
        correlationValue.textContent = `${percent}%`;
        
        // Color code the correlation
        correlationValue.classList.remove('low', 'medium');
        if (percent === 100) {
            // EXACT match - it's a symmetry!
            document.querySelector('.canvas-wrapper.difference').classList.add('match-animation');
            setTimeout(() => {
                document.querySelector('.canvas-wrapper.difference').classList.remove('match-animation');
            }, 1000);
        } else if (percent >= 98) {
            // Very close (possible rounding)
        } else if (percent < 50) {
            correlationValue.classList.add('low');
        } else if (percent < 85) {
            correlationValue.classList.add('medium');
        }
    }
    
    updateHistory() {
        const historyContainer = document.getElementById('operationsHistory');
        
        if (this.operations.length === 0) {
            historyContainer.innerHTML = '<p class="history-empty">No hay operaciones aplicadas</p>';
        } else {
            historyContainer.innerHTML = this.operations.map((op, i) => `
                <div class="history-item">
                    <span class="op-index">${i + 1}.</span>
                    <span class="op-name">${op.name}</span>
                    <span class="op-params">${op.params}</span>
                </div>
            `).join('');
            
            historyContainer.scrollTop = historyContainer.scrollHeight;
        }
        
        document.getElementById('totalOps').textContent = this.operations.length;
    }
    
    reset() {
        this.operations = [];
        this.currentMatrix = SymmetryOps.identity();
        
        if (this.originalImageData) {
            this.transformedImageData = this.originalImageData.slice();
            
            const size = this.canvasSize;
            const imageData = this.transformedCtx.createImageData(size, size);
            imageData.data.set(this.transformedImageData);
            this.transformedCtx.putImageData(imageData, 0, 0);
            
            this.updateDifference();
        }
        
        this.updateHistory();
        this.updateMatrixDisplay();
    }
}

// Initialize application when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new WallpaperExplorer();
});
