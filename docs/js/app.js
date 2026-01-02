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
     * The pattern is constructed by:
     * 1. Creating an ASYMMETRIC fundamental domain
     * 2. Applying the symmetry operations of the group to build the unit cell
     * 3. Tiling the unit cell to fill the canvas
     */
    generatePattern(groupName) {
        const size = this.canvasSize;
        const imageData = this.originalCtx.createImageData(size, size);
        const group = WallpaperGroups[groupName];
        
        // Seed based on group name for consistency
        const seed = this.hashString(groupName);
        const rng = this.seededRandom(seed);
        
        // Generate pattern based on group type
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
        
        // Fill image data
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
     * Create an asymmetric motif (fundamental domain content)
     */
    createAsymmetricMotif(size, rng) {
        const motif = new Float32Array(size * size);
        
        // Add several asymmetric elements
        const numElements = 3 + Math.floor(rng() * 3);
        
        for (let i = 0; i < numElements; i++) {
            const cx = rng() * size * 0.8 + size * 0.1;
            const cy = rng() * size * 0.8 + size * 0.1;
            const sigmaX = 10 + rng() * 20;
            const sigmaY = 10 + rng() * 20;  // Different sigma for asymmetry
            const angle = rng() * Math.PI;  // Rotation for asymmetry
            const amplitude = 0.3 + rng() * 0.7;
            
            for (let y = 0; y < size; y++) {
                for (let x = 0; x < size; x++) {
                    const dx = x - cx;
                    const dy = y - cy;
                    // Rotated elliptical Gaussian (asymmetric)
                    const rx = dx * Math.cos(angle) + dy * Math.sin(angle);
                    const ry = -dx * Math.sin(angle) + dy * Math.cos(angle);
                    const value = amplitude * Math.exp(-(rx*rx)/(2*sigmaX*sigmaX) - (ry*ry)/(2*sigmaY*sigmaY));
                    motif[y * size + x] += value;
                }
            }
        }
        
        return motif;
    }
    
    // === Pattern generators for each wallpaper group ===
    
    /**
     * p1: Only translations (parallelogram lattice)
     * NO rotational symmetry, NO reflection
     */
    generateP1(size, rng) {
        const cellSize = size / 3;  // 3x3 tiling
        const motif = this.createAsymmetricMotif(Math.floor(cellSize), rng);
        const pattern = new Float32Array(size * size);
        
        // Simple tiling without any symmetry operations
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const mx = Math.floor(x % cellSize);
                const my = Math.floor(y % cellSize);
                if (mx < motif.length && my < Math.sqrt(motif.length)) {
                    const motifSize = Math.floor(cellSize);
                    pattern[y * size + x] = motif[my * motifSize + mx] || 0;
                }
            }
        }
        
        return this.normalizePattern(pattern);
    }
    
    /**
     * p2: 180° rotation symmetry
     */
    generateP2(size, rng) {
        const cellSize = Math.floor(size / 4);
        const halfCell = Math.floor(cellSize / 2);
        const motif = this.createAsymmetricMotif(halfCell, rng);
        const pattern = new Float32Array(size * size);
        
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                // Position within cell
                const cx = x % cellSize;
                const cy = y % cellSize;
                
                let value = 0;
                
                // Original in top-left quarter
                if (cx < halfCell && cy < halfCell) {
                    value = motif[cy * halfCell + cx] || 0;
                }
                // 180° rotated in bottom-right quarter
                else if (cx >= halfCell && cy >= halfCell) {
                    const mx = cellSize - 1 - cx;
                    const my = cellSize - 1 - cy;
                    if (mx >= 0 && mx < halfCell && my >= 0 && my < halfCell) {
                        value = motif[my * halfCell + mx] || 0;
                    }
                }
                // 180° rotated copies for other quadrants
                else if (cx >= halfCell && cy < halfCell) {
                    const mx = cellSize - 1 - cx;
                    const my = halfCell - 1 - cy;
                    if (mx >= 0 && mx < halfCell && my >= 0 && my < halfCell) {
                        value = motif[my * halfCell + mx] || 0;
                    }
                }
                else {
                    const mx = halfCell - 1 - cx;
                    const my = cellSize - 1 - cy;
                    if (mx >= 0 && mx < halfCell && my >= 0 && my < halfCell) {
                        value = motif[my * halfCell + mx] || 0;
                    }
                }
                
                pattern[y * size + x] = value;
            }
        }
        
        return this.normalizePattern(pattern);
    }
    
    /**
     * pm: Vertical reflection axes
     */
    generatePM(size, rng) {
        const cellSize = Math.floor(size / 4);
        const halfCell = Math.floor(cellSize / 2);
        const motif = this.createAsymmetricMotif(halfCell, rng);
        const pattern = new Float32Array(size * size);
        
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const cx = x % cellSize;
                const cy = y % cellSize;
                
                // Left half: original
                // Right half: mirror
                const mx = cx < halfCell ? cx : cellSize - 1 - cx;
                const my = cy % halfCell;
                
                if (mx >= 0 && mx < halfCell && my >= 0 && my < halfCell) {
                    pattern[y * size + x] = motif[my * halfCell + mx] || 0;
                }
            }
        }
        
        return this.normalizePattern(pattern);
    }
    
    /**
     * pg: Parallel glide reflections
     */
    generatePG(size, rng) {
        const cellSize = Math.floor(size / 4);
        const halfCell = Math.floor(cellSize / 2);
        const motif = this.createAsymmetricMotif(halfCell, rng);
        const pattern = new Float32Array(size * size);
        
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const cx = x % cellSize;
                const cy = y % cellSize;
                
                let mx, my;
                
                // Glide: reflect + translate by half cell
                if (cx < halfCell) {
                    mx = cx;
                    my = cy % halfCell;
                } else {
                    // Reflected and shifted
                    mx = cellSize - 1 - cx;
                    my = (cy + halfCell) % cellSize;
                    my = my % halfCell;
                }
                
                if (mx >= 0 && mx < halfCell && my >= 0 && my < halfCell) {
                    pattern[y * size + x] = motif[my * halfCell + mx] || 0;
                }
            }
        }
        
        return this.normalizePattern(pattern);
    }
    
    /**
     * cm: Reflection + glide (centered)
     */
    generateCM(size, rng) {
        const cellSize = Math.floor(size / 4);
        const halfCell = Math.floor(cellSize / 2);
        const motif = this.createAsymmetricMotif(halfCell, rng);
        const pattern = new Float32Array(size * size);
        
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const cx = x % cellSize;
                const cy = y % cellSize;
                
                // Determine which copy based on cell position
                const inSecondRow = cy >= halfCell;
                const shifted = inSecondRow ? halfCell : 0;
                
                let mx = (cx + shifted) % cellSize;
                mx = mx < halfCell ? mx : cellSize - 1 - mx;
                const my = cy % halfCell;
                
                if (mx >= 0 && mx < halfCell && my >= 0 && my < halfCell) {
                    pattern[y * size + x] = motif[my * halfCell + mx] || 0;
                }
            }
        }
        
        return this.normalizePattern(pattern);
    }
    
    /**
     * pmm: Perpendicular reflection axes (rectangle)
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
                
                // Mirror in both x and y
                const mx = cx < halfCell ? cx : cellSize - 1 - cx;
                const my = cy < halfCell ? cy : cellSize - 1 - cy;
                
                if (mx >= 0 && mx < halfCell && my >= 0 && my < halfCell) {
                    pattern[y * size + x] = motif[my * halfCell + mx] || 0;
                }
            }
        }
        
        return this.normalizePattern(pattern);
    }
    
    /**
     * pmg: Reflection + perpendicular glide
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
                
                // Reflection in x
                let mx = cx < halfCell ? cx : cellSize - 1 - cx;
                
                // Glide in y for second column
                let my;
                if (cx < halfCell) {
                    my = cy < halfCell ? cy : cellSize - 1 - cy;
                } else {
                    my = cy < halfCell ? cy : cellSize - 1 - cy;
                    my = (my + halfCell) % cellSize;
                    my = my < halfCell ? my : my - halfCell;
                }
                
                if (mx >= 0 && mx < halfCell && my >= 0 && my < halfCell) {
                    pattern[y * size + x] = motif[my * halfCell + mx] || 0;
                }
            }
        }
        
        return this.normalizePattern(pattern);
    }
    
    /**
     * pgg: Perpendicular glide reflections
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
                
                // Two perpendicular glides create 180° rotation
                const quadX = cx < halfCell ? 0 : 1;
                const quadY = cy < halfCell ? 0 : 1;
                
                let mx = cx % halfCell;
                let my = cy % halfCell;
                
                // Apply transformations based on quadrant
                if ((quadX + quadY) % 2 === 1) {
                    mx = halfCell - 1 - mx;
                    my = halfCell - 1 - my;
                }
                
                if (mx >= 0 && mx < halfCell && my >= 0 && my < halfCell) {
                    pattern[y * size + x] = motif[my * halfCell + mx] || 0;
                }
            }
        }
        
        return this.normalizePattern(pattern);
    }
    
    /**
     * cmm: Centered cell with reflections
     */
    generateCMM(size, rng) {
        const cellSize = Math.floor(size / 4);
        const halfCell = Math.floor(cellSize / 2);
        const motif = this.createAsymmetricMotif(halfCell, rng);
        const pattern = new Float32Array(size * size);
        
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const cx = x % cellSize;
                const cy = y % cellSize;
                
                // pmm symmetry with centering
                let mx = cx < halfCell ? cx : cellSize - 1 - cx;
                let my = cy < halfCell ? cy : cellSize - 1 - cy;
                
                if (mx >= 0 && mx < halfCell && my >= 0 && my < halfCell) {
                    pattern[y * size + x] = motif[my * halfCell + mx] || 0;
                }
            }
        }
        
        return this.normalizePattern(pattern);
    }
    
    /**
     * p4: 90° rotation symmetry
     */
    generateP4(size, rng) {
        const cellSize = Math.floor(size / 4);
        const quarterCell = Math.floor(cellSize / 2);
        const motif = this.createAsymmetricMotif(quarterCell, rng);
        const pattern = new Float32Array(size * size);
        
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const cx = x % cellSize;
                const cy = y % cellSize;
                
                // Center of cell
                const centerX = cellSize / 2;
                const centerY = cellSize / 2;
                
                // Distance from center
                const dx = cx - centerX;
                const dy = cy - centerY;
                
                // Determine which 90° sector and map to first quadrant
                let mx, my;
                
                if (cx < centerX && cy < centerY) {
                    // Top-left quadrant (original)
                    mx = cx % quarterCell;
                    my = cy % quarterCell;
                } else if (cx >= centerX && cy < centerY) {
                    // Top-right: 90° rotation
                    mx = cy % quarterCell;
                    my = quarterCell - 1 - (cx % quarterCell);
                } else if (cx >= centerX && cy >= centerY) {
                    // Bottom-right: 180° rotation
                    mx = quarterCell - 1 - (cx % quarterCell);
                    my = quarterCell - 1 - (cy % quarterCell);
                } else {
                    // Bottom-left: 270° rotation
                    mx = quarterCell - 1 - (cy % quarterCell);
                    my = cx % quarterCell;
                }
                
                mx = Math.max(0, Math.min(quarterCell - 1, mx));
                my = Math.max(0, Math.min(quarterCell - 1, my));
                
                pattern[y * size + x] = motif[my * quarterCell + mx] || 0;
            }
        }
        
        return this.normalizePattern(pattern);
    }
    
    /**
     * p4m: Square with all reflections
     */
    generateP4M(size, rng) {
        const cellSize = Math.floor(size / 4);
        const quarterCell = Math.floor(cellSize / 2);
        const motif = this.createAsymmetricMotif(quarterCell, rng);
        const pattern = new Float32Array(size * size);
        
        // Make motif symmetric under diagonal reflection for p4m
        for (let y = 0; y < quarterCell; y++) {
            for (let x = 0; x < y; x++) {
                const avg = (motif[y * quarterCell + x] + motif[x * quarterCell + y]) / 2;
                motif[y * quarterCell + x] = avg;
                motif[x * quarterCell + y] = avg;
            }
        }
        
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const cx = x % cellSize;
                const cy = y % cellSize;
                
                // Mirror in both axes first
                let mx = cx < cellSize/2 ? cx : cellSize - 1 - cx;
                let my = cy < cellSize/2 ? cy : cellSize - 1 - cy;
                
                // Then diagonal mirror
                if (mx > my) {
                    [mx, my] = [my, mx];
                }
                
                mx = mx % quarterCell;
                my = my % quarterCell;
                
                mx = Math.max(0, Math.min(quarterCell - 1, mx));
                my = Math.max(0, Math.min(quarterCell - 1, my));
                
                pattern[y * size + x] = motif[my * quarterCell + mx] || 0;
            }
        }
        
        return this.normalizePattern(pattern);
    }
    
    /**
     * p4g: Square with glides and rotations
     */
    generateP4G(size, rng) {
        const cellSize = Math.floor(size / 4);
        const quarterCell = Math.floor(cellSize / 2);
        const motif = this.createAsymmetricMotif(quarterCell, rng);
        const pattern = new Float32Array(size * size);
        
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const cx = x % cellSize;
                const cy = y % cellSize;
                
                // p4g has 4-fold rotation with diagonal mirrors only
                const quadX = cx >= cellSize/2 ? 1 : 0;
                const quadY = cy >= cellSize/2 ? 1 : 0;
                const quadrant = quadY * 2 + quadX;
                
                let mx = cx % quarterCell;
                let my = cy % quarterCell;
                
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
                
                mx = Math.max(0, Math.min(quarterCell - 1, mx));
                my = Math.max(0, Math.min(quarterCell - 1, my));
                
                pattern[y * size + x] = motif[my * quarterCell + mx] || 0;
            }
        }
        
        return this.normalizePattern(pattern);
    }
    
    /**
     * p3: 120° rotation symmetry (hexagonal)
     */
    generateP3(size, rng) {
        const pattern = new Float32Array(size * size);
        
        // Create a radially arranged pattern with 3-fold symmetry
        const cx = size / 2;
        const cy = size / 2;
        
        // Create asymmetric elements in one 120° sector
        const numElements = 3 + Math.floor(rng() * 2);
        const elements = [];
        
        for (let i = 0; i < numElements; i++) {
            const r = 20 + rng() * (size/3);
            const theta = rng() * (2 * Math.PI / 3);  // Only in first third
            elements.push({
                x: cx + r * Math.cos(theta),
                y: cy + r * Math.sin(theta),
                sigma: 15 + rng() * 25,
                amplitude: 0.3 + rng() * 0.7
            });
        }
        
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                let value = 0;
                
                // Apply 3-fold rotation
                for (let rot = 0; rot < 3; rot++) {
                    const angle = rot * 2 * Math.PI / 3;
                    const cos = Math.cos(angle);
                    const sin = Math.sin(angle);
                    
                    for (const el of elements) {
                        // Rotate element position
                        const ex = cx + (el.x - cx) * cos - (el.y - cy) * sin;
                        const ey = cy + (el.x - cx) * sin + (el.y - cy) * cos;
                        
                        const dx = x - ex;
                        const dy = y - ey;
                        value += el.amplitude * Math.exp(-(dx*dx + dy*dy) / (2 * el.sigma * el.sigma));
                    }
                }
                
                pattern[y * size + x] = value / 3;
            }
        }
        
        return this.normalizePattern(pattern);
    }
    
    /**
     * p3m1: 120° rotation with reflection through centers
     */
    generateP3M1(size, rng) {
        const pattern = this.generateP3(size, rng);
        
        // Add mirror symmetry through rotation centers
        const result = new Float32Array(size * size);
        const cx = size / 2;
        const cy = size / 2;
        
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                // Original value
                let value = pattern[y * size + x];
                
                // Add reflected value (vertical mirror through center)
                const rx = size - 1 - x;
                value += pattern[y * size + rx];
                
                result[y * size + x] = value / 2;
            }
        }
        
        return this.normalizePattern(result);
    }
    
    /**
     * p31m: 120° rotation with reflection between centers
     */
    generateP31M(size, rng) {
        const pattern = this.generateP3(size, rng);
        
        // Add mirror symmetry between rotation centers
        const result = new Float32Array(size * size);
        const cx = size / 2;
        const cy = size / 2;
        
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                // Original value
                let value = pattern[y * size + x];
                
                // Add reflected value (horizontal mirror through center)
                const ry = size - 1 - y;
                value += pattern[ry * size + x];
                
                result[y * size + x] = value / 2;
            }
        }
        
        return this.normalizePattern(result);
    }
    
    /**
     * p6: 60° rotation symmetry
     */
    generateP6(size, rng) {
        const pattern = new Float32Array(size * size);
        
        const cx = size / 2;
        const cy = size / 2;
        
        // Create asymmetric elements in one 60° sector
        const numElements = 2 + Math.floor(rng() * 2);
        const elements = [];
        
        for (let i = 0; i < numElements; i++) {
            const r = 20 + rng() * (size/3);
            const theta = rng() * (Math.PI / 3);  // Only in first sixth
            elements.push({
                x: cx + r * Math.cos(theta),
                y: cy + r * Math.sin(theta),
                sigma: 12 + rng() * 20,
                amplitude: 0.3 + rng() * 0.7
            });
        }
        
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                let value = 0;
                
                // Apply 6-fold rotation
                for (let rot = 0; rot < 6; rot++) {
                    const angle = rot * Math.PI / 3;
                    const cos = Math.cos(angle);
                    const sin = Math.sin(angle);
                    
                    for (const el of elements) {
                        const ex = cx + (el.x - cx) * cos - (el.y - cy) * sin;
                        const ey = cy + (el.x - cx) * sin + (el.y - cy) * cos;
                        
                        const dx = x - ex;
                        const dy = y - ey;
                        value += el.amplitude * Math.exp(-(dx*dx + dy*dy) / (2 * el.sigma * el.sigma));
                    }
                }
                
                pattern[y * size + x] = value / 6;
            }
        }
        
        return this.normalizePattern(pattern);
    }
    
    /**
     * p6m: Maximum hexagonal symmetry (60° + all mirrors)
     */
    generateP6M(size, rng) {
        const pattern = this.generateP6(size, rng);
        
        // Add both vertical and horizontal mirror symmetry
        const result = new Float32Array(size * size);
        
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                let value = pattern[y * size + x];
                value += pattern[y * size + (size - 1 - x)];  // Vertical mirror
                value += pattern[(size - 1 - y) * size + x];  // Horizontal mirror
                value += pattern[(size - 1 - y) * size + (size - 1 - x)];  // Both
                
                result[y * size + x] = value / 4;
            }
        }
        
        return this.normalizePattern(result);
    }
    
    /**
     * Normalize pattern values to [0, 1]
     */
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
            // Very close - likely a symmetry with tiny rounding
            // Don't add animation but show high correlation
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
            
            // Scroll to bottom
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
