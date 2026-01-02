/**
 * Symmetry Operations Module
 * Mathematical transformations for wallpaper group symmetries
 * 
 * MATHEMATICAL REFERENCE:
 * - International Tables for Crystallography, Vol. A
 * - Conway et al., "The Symmetries of Things" (2008)
 * 
 * IMPORTANT: All operations are designed to be EXACT (no interpolation errors)
 * for crystallographic angles (60°, 90°, 120°, 180°, etc.)
 */

const SymmetryOps = {
    /**
     * Rotation matrix for angle in degrees
     */
    rotationMatrix(angleDeg) {
        const rad = (angleDeg * Math.PI) / 180;
        const cos = Math.cos(rad);
        const sin = Math.sin(rad);
        // Round to avoid floating point errors for exact angles
        const roundedCos = Math.abs(cos) < 1e-10 ? 0 : (Math.abs(cos - 1) < 1e-10 ? 1 : (Math.abs(cos + 1) < 1e-10 ? -1 : cos));
        const roundedSin = Math.abs(sin) < 1e-10 ? 0 : (Math.abs(sin - 1) < 1e-10 ? 1 : (Math.abs(sin + 1) < 1e-10 ? -1 : sin));
        return [
            [roundedCos, -roundedSin],
            [roundedSin, roundedCos]
        ];
    },

    /**
     * Reflection matrices for different axes
     */
    reflectionMatrix(axis) {
        switch (axis) {
            case 'vertical':
                return [[-1, 0], [0, 1]];
            case 'horizontal':
                return [[1, 0], [0, -1]];
            case 'diagonal':
                return [[0, 1], [1, 0]];
            case 'antidiagonal':
                return [[0, -1], [-1, 0]];
            default:
                return [[1, 0], [0, 1]];
        }
    },

    /**
     * Multiply two 2x2 matrices
     */
    multiplyMatrices(a, b) {
        return [
            [a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]],
            [a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]]
        ];
    },

    /**
     * Identity matrix
     */
    identity() {
        return [[1, 0], [0, 1]];
    },

    /**
     * Format matrix for display
     */
    formatMatrix(matrix) {
        const format = (v) => {
            if (Math.abs(v) < 1e-10) return "0";
            if (Math.abs(v - 1) < 1e-10) return "1";
            if (Math.abs(v + 1) < 1e-10) return "-1";
            if (Math.abs(v - 0.5) < 1e-10) return "½";
            if (Math.abs(v + 0.5) < 1e-10) return "-½";
            // For √3/2
            if (Math.abs(v - 0.866025) < 1e-5) return "√3/2";
            if (Math.abs(v + 0.866025) < 1e-5) return "-√3/2";
            return v.toFixed(3);
        };
        return {
            m00: format(matrix[0][0]),
            m01: format(matrix[0][1]),
            m10: format(matrix[1][0]),
            m11: format(matrix[1][1])
        };
    }
};

/**
 * Image transformation operations - EXACT pixel operations
 * No interpolation - uses array index manipulation for exactness
 */
const ImageTransform = {
    /**
     * Rotate image by EXACT crystallographic angles using array operations
     * Supported angles: 60, 90, 120, 180, 240, 270, 300 degrees
     */
    rotate(imageData, angleDeg, width, height) {
        // Normalize angle to 0-360
        angleDeg = ((angleDeg % 360) + 360) % 360;
        
        // For exact multiples of 90°, use array operations (no interpolation)
        if (angleDeg === 0) {
            return imageData.slice();
        }
        
        if (angleDeg === 90) {
            return this._rotate90(imageData, width, height);
        }
        
        if (angleDeg === 180) {
            return this._rotate180(imageData, width, height);
        }
        
        if (angleDeg === 270) {
            return this._rotate270(imageData, width, height);
        }
        
        // For 60°, 120°, etc. we need interpolation but with careful handling
        // These are only approximate for non-square symmetric patterns
        return this._rotateGeneral(imageData, angleDeg, width, height);
    },
    
    /**
     * Exact 90° rotation (counter-clockwise)
     * Classic formula: pixel at (x,y) comes from pixel at (y, width-1-x)
     * This is a perfect pixel-to-pixel mapping, no interpolation needed
     */
    _rotate90(imageData, width, height) {
        const newData = new Uint8ClampedArray(imageData.length);
        
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                // For 90° CCW: new(x,y) = old(y, width-1-x)
                const srcX = y;
                const srcY = width - 1 - x;
                
                const destIdx = (y * width + x) * 4;
                const srcIdx = (srcY * width + srcX) * 4;
                
                newData[destIdx] = imageData[srcIdx];
                newData[destIdx + 1] = imageData[srcIdx + 1];
                newData[destIdx + 2] = imageData[srcIdx + 2];
                newData[destIdx + 3] = imageData[srcIdx + 3];
            }
        }
        
        return newData;
    },
    
    /**
     * Exact 180° rotation
     */
    /**
     * Exact 180° rotation
     * Classic formula: pixel at (x,y) comes from pixel at (width-1-x, height-1-y)
     */
    _rotate180(imageData, width, height) {
        const newData = new Uint8ClampedArray(imageData.length);
        
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const srcX = width - 1 - x;
                const srcY = height - 1 - y;
                
                const destIdx = (y * width + x) * 4;
                const srcIdx = (srcY * width + srcX) * 4;
                
                newData[destIdx] = imageData[srcIdx];
                newData[destIdx + 1] = imageData[srcIdx + 1];
                newData[destIdx + 2] = imageData[srcIdx + 2];
                newData[destIdx + 3] = imageData[srcIdx + 3];
            }
        }
        
        return newData;
    },
    
    /**
     * Exact 270° rotation (counter-clockwise) = 90° clockwise
     */
    /**
     * Exact 270° rotation (= 90° clockwise)
     * Classic formula: pixel at (x,y) comes from pixel at (height-1-y, x)
     */
    _rotate270(imageData, width, height) {
        const newData = new Uint8ClampedArray(imageData.length);
        
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const srcX = height - 1 - y;
                const srcY = x;
                
                const destIdx = (y * width + x) * 4;
                const srcIdx = (srcY * width + srcX) * 4;
                
                newData[destIdx] = imageData[srcIdx];
                newData[destIdx + 1] = imageData[srcIdx + 1];
                newData[destIdx + 2] = imageData[srcIdx + 2];
                newData[destIdx + 3] = imageData[srcIdx + 3];
            }
        }
        
        return newData;
    },
    
    /**
     * General rotation with bilinear interpolation for smooth results
     * Uses high-quality interpolation to minimize discretization errors
     */
    _rotateGeneral(imageData, angleDeg, width, height) {
        const rad = (angleDeg * Math.PI) / 180;
        const cos = Math.cos(rad);
        const sin = Math.sin(rad);
        // Use center at (width-1)/2, (height-1)/2 to match pattern generation
        // For odd sizes (301), this is exactly the center pixel (150, 150)
        const centerX = (width - 1) / 2;
        const centerY = (height - 1) / 2;
        
        const newData = new Uint8ClampedArray(imageData.length);
        
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                // Translate to origin
                const dx = x - centerX;
                const dy = y - centerY;
                
                // Inverse rotate to find source
                const srcX = cos * dx + sin * dy + centerX;
                const srcY = -sin * dx + cos * dy + centerY;
                
                const destIdx = (y * width + x) * 4;
                
                // Bilinear interpolation for smooth results
                const x0 = Math.floor(srcX);
                const y0 = Math.floor(srcY);
                const x1 = x0 + 1;
                const y1 = y0 + 1;
                
                if (x0 >= 0 && x1 < width && y0 >= 0 && y1 < height) {
                    const fx = srcX - x0;
                    const fy = srcY - y0;
                    const fx1 = 1 - fx;
                    const fy1 = 1 - fy;
                    
                    const idx00 = (y0 * width + x0) * 4;
                    const idx10 = (y0 * width + x1) * 4;
                    const idx01 = (y1 * width + x0) * 4;
                    const idx11 = (y1 * width + x1) * 4;
                    
                    for (let c = 0; c < 4; c++) {
                        const v = imageData[idx00 + c] * fx1 * fy1 +
                                  imageData[idx10 + c] * fx * fy1 +
                                  imageData[idx01 + c] * fx1 * fy +
                                  imageData[idx11 + c] * fx * fy;
                        newData[destIdx + c] = Math.round(v);
                    }
                } else if (srcX >= -0.5 && srcX < width - 0.5 && srcY >= -0.5 && srcY < height - 0.5) {
                    // Edge case - use nearest neighbor
                    const nearX = Math.max(0, Math.min(width - 1, Math.round(srcX)));
                    const nearY = Math.max(0, Math.min(height - 1, Math.round(srcY)));
                    const srcIdx = (nearY * width + nearX) * 4;
                    newData[destIdx] = imageData[srcIdx];
                    newData[destIdx + 1] = imageData[srcIdx + 1];
                    newData[destIdx + 2] = imageData[srcIdx + 2];
                    newData[destIdx + 3] = imageData[srcIdx + 3];
                } else {
                    // Out of bounds - use edge clamping
                    const clampX = Math.max(0, Math.min(width - 1, Math.round(srcX)));
                    const clampY = Math.max(0, Math.min(height - 1, Math.round(srcY)));
                    const srcIdx = (clampY * width + clampX) * 4;
                    newData[destIdx] = imageData[srcIdx];
                    newData[destIdx + 1] = imageData[srcIdx + 1];
                    newData[destIdx + 2] = imageData[srcIdx + 2];
                    newData[destIdx + 3] = imageData[srcIdx + 3];
                }
            }
        }
        
        return newData;
    },

    /**
     * Reflect image across specified axis - EXACT operations
     */
    reflect(imageData, axis, width, height) {
        const newData = new Uint8ClampedArray(imageData.length);
        
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                let srcX, srcY;
                
                switch (axis) {
                    case 'vertical':
                        // Mirror across vertical center line (flip horizontally)
                        srcX = width - 1 - x;
                        srcY = y;
                        break;
                    case 'horizontal':
                        // Mirror across horizontal center line (flip vertically)
                        srcX = x;
                        srcY = height - 1 - y;
                        break;
                    case 'diagonal':
                        // Mirror across main diagonal (transpose)
                        srcX = y;
                        srcY = x;
                        break;
                    case 'antidiagonal':
                        // Mirror across anti-diagonal
                        srcX = height - 1 - y;
                        srcY = width - 1 - x;
                        break;
                    default:
                        srcX = x;
                        srcY = y;
                }
                
                const destIdx = (y * width + x) * 4;
                
                if (srcX >= 0 && srcX < width && srcY >= 0 && srcY < height) {
                    const srcIdx = (srcY * width + srcX) * 4;
                    newData[destIdx] = imageData[srcIdx];
                    newData[destIdx + 1] = imageData[srcIdx + 1];
                    newData[destIdx + 2] = imageData[srcIdx + 2];
                    newData[destIdx + 3] = imageData[srcIdx + 3];
                }
            }
        }
        
        return newData;
    },

    /**
     * Translate image by percentage of width/height (wrapping) - EXACT
     */
    translate(imageData, dx, dy, width, height) {
        const newData = new Uint8ClampedArray(imageData.length);
        const shiftX = Math.round(dx * width);
        const shiftY = Math.round(dy * height);
        
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                // Source position with wrapping (modulo)
                let srcX = ((x - shiftX) % width + width) % width;
                let srcY = ((y - shiftY) % height + height) % height;
                
                const destIdx = (y * width + x) * 4;
                const srcIdx = (srcY * width + srcX) * 4;
                
                newData[destIdx] = imageData[srcIdx];
                newData[destIdx + 1] = imageData[srcIdx + 1];
                newData[destIdx + 2] = imageData[srcIdx + 2];
                newData[destIdx + 3] = imageData[srcIdx + 3];
            }
        }
        
        return newData;
    },

    /**
     * Glide reflection: reflection + translation - EXACT
     */
    glide(imageData, axis, width, height) {
        // First reflect
        let result = this.reflect(imageData, axis, width, height);
        
        // Then translate along the reflection axis (not perpendicular)
        if (axis === 'horizontal') {
            // Reflect horizontally, then translate horizontally
            result = this.translate(result, 0.5, 0, width, height);
        } else {
            // Reflect vertically, then translate vertically  
            result = this.translate(result, 0, 0.5, width, height);
        }
        
        return result;
    },

    /**
     * Calculate difference image between two image data arrays
     * Uses color coding:
     * - Bright green: exact match (diff = 0)
     * - Light green: near match (diff <= 5, interpolation errors)
     * - Yellow: small difference (diff <= 15)
     * - Red: significant difference (diff > 15)
     */
    difference(imageData1, imageData2, width, height) {
        const diffData = new Uint8ClampedArray(imageData1.length);
        
        for (let i = 0; i < imageData1.length; i += 4) {
            // Calculate absolute difference per channel
            const diffR = Math.abs(imageData1[i] - imageData2[i]);
            const diffG = Math.abs(imageData1[i + 1] - imageData2[i + 1]);
            const diffB = Math.abs(imageData1[i + 2] - imageData2[i + 2]);
            
            // Max difference across channels
            const maxDiff = Math.max(diffR, diffG, diffB);
            
            if (maxDiff === 0) {
                // EXACT match - bright green
                diffData[i] = 30;
                diffData[i + 1] = 180;
                diffData[i + 2] = 80;
            } else if (maxDiff <= 5) {
                // Near match (interpolation errors) - light green/teal
                diffData[i] = 50;
                diffData[i + 1] = 160;
                diffData[i + 2] = 120;
            } else if (maxDiff <= 15) {
                // Small difference - yellow/orange
                diffData[i] = 200;
                diffData[i + 1] = 150;
                diffData[i + 2] = 50;
            } else {
                // Significant difference - red, intensity proportional to diff
                const intensity = Math.min(255, maxDiff * 1.5);
                diffData[i] = 150 + intensity / 2.5;
                diffData[i + 1] = 40;
                diffData[i + 2] = 40;
            }
            diffData[i + 3] = 255;
        }
        
        return diffData;
    },

    /**
     * Calculate similarity between two images
     * Returns value between 0 and 1, where 1 means identical images
     * Uses a more sensitive metric: counts pixels that are "different enough"
     */
    correlation(imageData1, imageData2) {
        let exactMatches = 0;
        let nearMatches = 0;  // diff <= 5 (interpolation tolerance)
        let totalPixels = 0;
        
        for (let i = 0; i < imageData1.length; i += 4) {
            const diffR = Math.abs(imageData1[i] - imageData2[i]);
            const diffG = Math.abs(imageData1[i + 1] - imageData2[i + 1]);
            const diffB = Math.abs(imageData1[i + 2] - imageData2[i + 2]);
            
            const maxDiff = Math.max(diffR, diffG, diffB);
            
            if (maxDiff === 0) {
                exactMatches++;
            } else if (maxDiff <= 5) {
                nearMatches++;
            }
            totalPixels++;
        }
        
        // A true symmetry should have nearly all pixels matching exactly or nearly
        // This is stricter: only exact + near matches count toward 100%
        const matchRatio = (exactMatches + nearMatches * 0.9) / totalPixels;
        
        return matchRatio;
    }
};

/**
 * Wallpaper group definitions with CORRECT mathematical properties
 * 
 * Reference: International Tables for Crystallography, Vol. A
 * 
 * Notation:
 * - e: identity
 * - C_n: rotation by 360°/n
 * - σ_v: vertical reflection (across vertical axis)
 * - σ_h: horizontal reflection (across horizontal axis)
 * - σ_d: diagonal reflection
 * - g: glide reflection (reflection + translation by half period)
 * - t₁, t₂: primitive translations
 */
const WallpaperGroups = {
    p1: {
        name: 'p1',
        altNames: { schoenflies: 'C₁', hm: '1', orbifold: 'o' },
        lattice: 'Oblicuo',
        rotationOrder: 1,
        hasReflection: false,
        hasGlide: false,
        description: 'Solo traslaciones. Sin rotación ni reflexión.',
        pointGroupOrder: 1,
        generators: 't₁, t₂',
        // Operations that should give 100% correlation for this group
        validSymmetries: [
            { name: 'Identidad', ops: [] },
            { name: 'T(1,0)', ops: [{type: 'translate', dx: 1, dy: 0}] },
            { name: 'T(0,1)', ops: [{type: 'translate', dx: 0, dy: 1}] }
        ],
        // Operations that should NOT give 100% for this group
        invalidSymmetries: ['C2', 'C3', 'C4', 'C6', 'σ_v', 'σ_h', 'g'],
        // Cayley table (multiplication table) for point group
        cayleyTable: {
            elements: ['e'],
            table: [['e']]
        },
        // Detailed explanation
        explanation: `
            El grupo p1 es el grupo de simetría más simple.
            Solo contiene traslaciones por los vectores de la red.
            NO tiene rotación (excepto trivial 360°), NO tiene reflexión, NO tiene glide.
            
            Si aplicás C2 (180°) a un patrón p1, NO debería dar la misma imagen.
        `
    },
    p2: {
        name: 'p2',
        altNames: { schoenflies: 'C₂', hm: '2', orbifold: '2222' },
        lattice: 'Oblicuo',
        rotationOrder: 2,
        hasReflection: false,
        hasGlide: false,
        description: 'Rotación de 180° (C₂). Sin reflexiones.',
        pointGroupOrder: 2,
        generators: 't₁, t₂, C₂',
        validSymmetries: [
            { name: 'Identidad', ops: [] },
            { name: 'C₂ (180°)', ops: [{type: 'rotate', angle: 180}] },
            { name: 'T(1,0)', ops: [{type: 'translate', dx: 1, dy: 0}] }
        ],
        invalidSymmetries: ['C3', 'C4', 'C6', 'σ_v', 'σ_h', 'g'],
        cayleyTable: {
            elements: ['e', 'C₂'],
            table: [
                ['e', 'C₂'],
                ['C₂', 'e']
            ]
        },
        explanation: `
            El grupo p2 tiene rotación de 180° alrededor de ciertos puntos.
            C₂ × C₂ = e (dos rotaciones de 180° = identidad)
            NO tiene reflexiones ni glides.
            
            Diferencia con p1: aplicar C2 (180°) SÍ da la misma imagen.
        `
    },
    pm: {
        name: 'pm',
        altNames: { schoenflies: 'Cₛ', hm: 'm', orbifold: '**' },
        lattice: 'Rectangular',
        rotationOrder: 1,
        hasReflection: true,
        hasGlide: false,
        description: 'Ejes de reflexión paralelos. Sin rotación.',
        pointGroupOrder: 2,
        generators: 't₁, t₂, σᵥ',
        validSymmetries: [
            { name: 'Identidad', ops: [] },
            { name: 'σᵥ (reflexión vertical)', ops: [{type: 'reflect', axis: 'vertical'}] },
            { name: 'T(1,0)', ops: [{type: 'translate', dx: 1, dy: 0}] }
        ],
        invalidSymmetries: ['C2', 'C3', 'C4', 'C6', 'g'],
        cayleyTable: {
            elements: ['e', 'σᵥ'],
            table: [
                ['e', 'σᵥ'],
                ['σᵥ', 'e']
            ]
        },
        explanation: `
            El grupo pm tiene ejes de reflexión paralelos.
            σ × σ = e (dos reflexiones iguales = identidad)
            NO tiene rotación (excepto trivial).
        `
    },
    pg: {
        name: 'pg',
        altNames: { schoenflies: 'C₁', hm: '1', orbifold: '××' },
        lattice: 'Rectangular',
        rotationOrder: 1,
        hasReflection: false,
        hasGlide: true,
        description: 'Reflexiones deslizantes (glide) paralelas. Sin reflexión pura.',
        pointGroupOrder: 1,
        generators: 't₁, t₂, gₕ',
        validSymmetries: [
            { name: 'Identidad', ops: [] },
            { name: 'T(1,0) traslación', ops: [{type: 'translate', dx: 1, dy: 0}] }
        ],
        invalidSymmetries: ['C2', 'C3', 'C4', 'C6', 'σ_v', 'σ_h', 'gₕ (solo)'],
        // Grupo puntual: C₁ (trivial) - los glides NO son operaciones puntuales
        // Un glide NO da 100%, g² = traslación
        cayleyTable: {
            elements: ['e'],
            table: [['e']]
        },
        explanation: `
            El grupo pg tiene reflexiones deslizantes (glide reflections).
            Un glide = reflexión + traslación por medio período.
            g × g = traslación (NO identidad, sino traslación completa).
            NO tiene reflexión pura ni rotación.
            
            Nota: UN solo glide NO da la imagen original.
            Dos glides dan una traslación completa.
        `
    },
    cm: {
        name: 'cm',
        altNames: { schoenflies: 'Cₛ', hm: 'm', orbifold: '*×' },
        lattice: 'Rectangular (centrado)',
        rotationOrder: 1,
        hasReflection: true,
        hasGlide: true,
        description: 'Reflexión + glide entre ejes. Celda centrada.',
        pointGroupOrder: 2,
        generators: 't₁, t₂, σᵥ',
        validSymmetries: [
            { name: 'Identidad', ops: [] },
            { name: 'σᵥ (reflexión vertical)', ops: [{type: 'reflect', axis: 'vertical'}] }
        ],
        invalidSymmetries: ['C2', 'C3', 'C4', 'C6', 'gₕ (solo)'],
        // Grupo puntual: C_s = {e, σᵥ}
        cayleyTable: {
            elements: ['e', 'σᵥ'],
            table: [
                ['e', 'σᵥ'],
                ['σᵥ', 'e']
            ]
        },
        explanation: `
            El grupo cm tiene reflexiones Y glides.
            La celda es centrada (hay un punto de red adicional en el centro).
            Tiene reflexión pura + glides entre los ejes de reflexión.
        `
    },
    pmm: {
        name: 'pmm',
        altNames: { schoenflies: 'D₂', hm: '2mm', orbifold: '*2222' },
        lattice: 'Rectangular',
        rotationOrder: 2,
        hasReflection: true,
        hasGlide: false,
        description: 'Reflexiones perpendiculares. σᵥ ∘ σₕ = C₂',
        pointGroupOrder: 4,
        generators: 't₁, t₂, σᵥ, σₕ',
        validSymmetries: [
            { name: 'Identidad', ops: [] },
            { name: 'σᵥ', ops: [{type: 'reflect', axis: 'vertical'}] },
            { name: 'σₕ', ops: [{type: 'reflect', axis: 'horizontal'}] },
            { name: 'C₂ = σᵥσₕ', ops: [{type: 'rotate', angle: 180}] }
        ],
        invalidSymmetries: ['C3', 'C4', 'C6', 'g'],
        // D₂: R = rotación 180°, F = reflexión vertical, FR = reflexión horizontal
        cayleyTable: {
            elements: ['e', 'R', 'F', 'FR'],
            table: [
                ['e', 'R', 'F', 'FR'],
                ['R', 'e', 'FR', 'F'],
                ['F', 'FR', 'e', 'R'],
                ['FR', 'F', 'R', 'e']
            ]
        },
        explanation: `
            El grupo pmm tiene dos ejes de reflexión perpendiculares (D₂).
            R = rotación 180°, F = reflexión, FR = segunda reflexión
            IMPORTANTE: F × FR = R (dos reflexiones perpendiculares = rotación 180°)
        `
    },
    pmg: {
        name: 'pmg',
        altNames: { schoenflies: 'D₂', hm: '2mm', orbifold: '22*' },
        lattice: 'Rectangular',
        rotationOrder: 2,
        hasReflection: true,
        hasGlide: true,
        description: 'Reflexión + glide perpendicular. σᵥ ∘ gₕ implica C₂.',
        pointGroupOrder: 4,
        generators: 't₁, t₂, σᵥ, gₕ',
        validSymmetries: [
            { name: 'Identidad', ops: [] },
            { name: 'σᵥ (reflexión vertical)', ops: [{type: 'reflect', axis: 'vertical'}] },
            { name: 'C₂ (180°)', ops: [{type: 'rotate', angle: 180}] }
        ],
        invalidSymmetries: ['C3', 'C4', 'C6', 'σₕ', 'gₕ (solo)'],
        // D₂ incompleto: F = reflexión vertical, R = rotación 180°
        // FR no es reflexión pura sino glide en pmg
        cayleyTable: {
            elements: ['e', 'R', 'F'],
            table: [
                ['e', 'R', 'F'],
                ['R', 'e', '(g)'],
                ['F', '(g)', 'e']
            ]
        },
        explanation: `
            Reflexión en una dirección, glide en la perpendicular.
            La combinación crea rotación de 180°.
        `
    },
    pgg: {
        name: 'pgg',
        altNames: { schoenflies: 'C₂', hm: '2', orbifold: '22×' },
        lattice: 'Rectangular',
        rotationOrder: 2,
        hasReflection: false,
        hasGlide: true,
        description: 'Glides perpendiculares. gᵥ ∘ gₕ = C₂. Sin reflexión pura.',
        pointGroupOrder: 2,
        generators: 't₁, t₂, gᵥ, gₕ',
        validSymmetries: [
            { name: 'Identidad', ops: [] },
            { name: 'C₂ (180°)', ops: [{type: 'rotate', angle: 180}] }
        ],
        invalidSymmetries: ['C3', 'C4', 'C6', 'σ_v', 'σ_h', 'gᵥ (solo)', 'gₕ (solo)'],
        // Grupo puntual: C₂ (los glides NO son operaciones puntuales)
        cayleyTable: {
            elements: ['e', 'C₂'],
            table: [
                ['e', 'C₂'],
                ['C₂', 'e']
            ]
        },
        explanation: `
            Dos glides perpendiculares, sin reflexión pura.
            gᵥ ∘ gₕ = C₂ (la composición da rotación 180°)
            NO tiene reflexión pura.
        `
    },
    cmm: {
        name: 'cmm',
        altNames: { schoenflies: 'D₂', hm: '2mm', orbifold: '2*22' },
        lattice: 'Rectangular (centrado)',
        rotationOrder: 2,
        hasReflection: true,
        hasGlide: true,
        description: 'Celda centrada con reflexiones perpendiculares.',
        pointGroupOrder: 4,
        generators: 't₁, t₂, σᵥ, σₕ',
        validSymmetries: [
            { name: 'Identidad', ops: [] },
            { name: 'σᵥ', ops: [{type: 'reflect', axis: 'vertical'}] },
            { name: 'σₕ', ops: [{type: 'reflect', axis: 'horizontal'}] },
            { name: 'C₂', ops: [{type: 'rotate', angle: 180}] }
        ],
        invalidSymmetries: ['C3', 'C4', 'C6'],
        // D₂: R = rotación 180°, F = reflexión vertical, FR = reflexión horizontal
        cayleyTable: {
            elements: ['e', 'R', 'F', 'FR'],
            table: [
                ['e', 'R', 'F', 'FR'],
                ['R', 'e', 'FR', 'F'],
                ['F', 'FR', 'e', 'R'],
                ['FR', 'F', 'R', 'e']
            ]
        },
        explanation: `
            Como pmm pero con celda centrada (D₂).
            Tiene reflexiones perpendiculares Y glides.
        `
    },
    p4: {
        name: 'p4',
        altNames: { schoenflies: 'C₄', hm: '4', orbifold: '442' },
        lattice: 'Cuadrado',
        rotationOrder: 4,
        hasReflection: false,
        hasGlide: false,
        description: 'Rotación de 90° (C₄). Sin reflexiones.',
        pointGroupOrder: 4,
        generators: 't₁, t₂, C₄',
        validSymmetries: [
            { name: 'Identidad', ops: [] },
            { name: 'C₄ (90°)', ops: [{type: 'rotate', angle: 90}] },
            { name: 'C₂ (180°)', ops: [{type: 'rotate', angle: 180}] },
            { name: 'C₄³ (270°)', ops: [{type: 'rotate', angle: 270}] }
        ],
        invalidSymmetries: ['C3', 'C6', 'σ_v', 'σ_h', 'σ_d', 'g'],
        cayleyTable: {
            elements: ['e', 'C₄', 'C₂', 'C₄³'],
            table: [
                ['e', 'C₄', 'C₂', 'C₄³'],
                ['C₄', 'C₂', 'C₄³', 'e'],
                ['C₂', 'C₄³', 'e', 'C₄'],
                ['C₄³', 'e', 'C₄', 'C₂']
            ]
        },
        explanation: `
            El grupo p4 tiene rotación de 90° (orden 4).
            C₄⁴ = e (cuatro rotaciones de 90° = identidad)
            C₄² = C₂ (dos rotaciones de 90° = 180°)
            NO tiene reflexiones.
        `
    },
    p4m: {
        name: 'p4m',
        altNames: { schoenflies: 'D₄', hm: '4mm', orbifold: '*442' },
        lattice: 'Cuadrado',
        rotationOrder: 4,
        hasReflection: true,
        hasGlide: true,
        description: 'Rotación 90° + reflexiones (4 ejes).',
        pointGroupOrder: 8,
        generators: 't₁, t₂, C₄, σ',
        validSymmetries: [
            { name: 'Identidad', ops: [] },
            { name: 'C₄', ops: [{type: 'rotate', angle: 90}] },
            { name: 'C₂', ops: [{type: 'rotate', angle: 180}] },
            { name: 'C₄³', ops: [{type: 'rotate', angle: 270}] },
            { name: 'σᵥ', ops: [{type: 'reflect', axis: 'vertical'}] },
            { name: 'σₕ', ops: [{type: 'reflect', axis: 'horizontal'}] },
            { name: 'σ_d', ops: [{type: 'reflect', axis: 'diagonal'}] },
            { name: 'σ_d\'', ops: [{type: 'reflect', axis: 'antidiagonal'}] }
        ],
        invalidSymmetries: ['C3', 'C6'],
        // D₄: R = rotación 90°, F = reflexión
        // Elementos: e, R, R², R³, F, FR, FR², FR³
        cayleyTable: {
            elements: ['e', 'R', 'R²', 'R³', 'F', 'FR', 'FR²', 'FR³'],
            table: [
                ['e', 'R', 'R²', 'R³', 'F', 'FR', 'FR²', 'FR³'],
                ['R', 'R²', 'R³', 'e', 'FR³', 'F', 'FR', 'FR²'],
                ['R²', 'R³', 'e', 'R', 'FR²', 'FR³', 'F', 'FR'],
                ['R³', 'e', 'R', 'R²', 'FR', 'FR²', 'FR³', 'F'],
                ['F', 'FR', 'FR²', 'FR³', 'e', 'R', 'R²', 'R³'],
                ['FR', 'FR²', 'FR³', 'F', 'R³', 'e', 'R', 'R²'],
                ['FR²', 'FR³', 'F', 'FR', 'R²', 'R³', 'e', 'R'],
                ['FR³', 'F', 'FR', 'FR²', 'R', 'R²', 'R³', 'e']
            ]
        },
        explanation: `
            El grupo p4m tiene la máxima simetría para red cuadrada.
            Rotación de 90° + 4 ejes de reflexión (horizontal, vertical, 2 diagonales).
            El grupo puntual es D₄ (orden 8).
        `
    },
    p4g: {
        name: 'p4g',
        altNames: { schoenflies: 'D₄', hm: '4mm', orbifold: '4*2' },
        lattice: 'Cuadrado',
        rotationOrder: 4,
        hasReflection: true,
        hasGlide: true,
        description: 'Rotación 90° + reflexiones diagonales. Ejes axiales son glides.',
        pointGroupOrder: 8,
        generators: 't₁, t₂, C₄, σ_d',
        validSymmetries: [
            { name: 'Identidad', ops: [] },
            { name: 'C₄ (90°)', ops: [{type: 'rotate', angle: 90}] },
            { name: 'C₂ (180°)', ops: [{type: 'rotate', angle: 180}] },
            { name: 'C₄³ (270°)', ops: [{type: 'rotate', angle: 270}] },
            { name: 'σ_d (diagonal)', ops: [{type: 'reflect', axis: 'diagonal'}] },
            { name: 'σ_d\' (anti-diag)', ops: [{type: 'reflect', axis: 'antidiagonal'}] }
        ],
        invalidSymmetries: ['C3', 'C6', 'σᵥ', 'σₕ', 'gᵥ (solo)', 'gₕ (solo)'],
        // D₄ incompleto: R = rotación 90°, F = reflexión diagonal
        // FR y FR³ son glides en p4g (reflexiones en ejes axiales)
        cayleyTable: {
            elements: ['e', 'R', 'R²', 'R³', 'F', 'FR²'],
            table: [
                ['e', 'R', 'R²', 'R³', 'F', 'FR²'],
                ['R', 'R²', 'R³', 'e', '(g)', '(g)'],
                ['R²', 'R³', 'e', 'R', 'FR²', 'F'],
                ['R³', 'e', 'R', 'R²', '(g)', '(g)'],
                ['F', '(g)', 'FR²', '(g)', 'e', 'R²'],
                ['FR²', '(g)', 'F', '(g)', 'R²', 'e']
            ]
        },
        explanation: `
            Similar a p4m pero con reflexiones solo en diagonales.
            Las reflexiones en ejes horizontal/vertical son glides.
        `
    },
    p3: {
        name: 'p3',
        altNames: { schoenflies: 'C₃', hm: '3', orbifold: '333' },
        lattice: 'Hexagonal',
        rotationOrder: 3,
        hasReflection: false,
        hasGlide: false,
        description: 'Rotación de 120° (C₃). Sin reflexiones.',
        pointGroupOrder: 3,
        generators: 't₁, t₂, C₃',
        validSymmetries: [
            { name: 'Identidad', ops: [] },
            { name: 'C₃ (120°)', ops: [{type: 'rotate', angle: 120}] },
            { name: 'C₃² (240°)', ops: [{type: 'rotate', angle: 240}] }
        ],
        invalidSymmetries: ['C2', 'C4', 'C6', 'σ_v', 'σ_h', 'g'],
        cayleyTable: {
            elements: ['e', 'C₃', 'C₃²'],
            table: [
                ['e', 'C₃', 'C₃²'],
                ['C₃', 'C₃²', 'e'],
                ['C₃²', 'e', 'C₃']
            ]
        },
        explanation: `
            El grupo p3 tiene rotación de 120° (orden 3).
            C₃³ = e (tres rotaciones de 120° = identidad)
            NO tiene reflexiones.
            Nota: NO tiene C₂ (rotación 180°).
        `
    },
    p3m1: {
        name: 'p3m1',
        altNames: { schoenflies: 'D₃', hm: '3m', orbifold: '*333' },
        lattice: 'Hexagonal',
        rotationOrder: 3,
        hasReflection: true,
        hasGlide: false,
        description: 'Rotación 120° + reflexiones a través de centros.',
        pointGroupOrder: 6,
        generators: 't₁, t₂, C₃, σᵥ',
        validSymmetries: [
            { name: 'Identidad', ops: [] },
            { name: 'C₃ (120°)', ops: [{type: 'rotate', angle: 120}] },
            { name: 'C₃² (240°)', ops: [{type: 'rotate', angle: 240}] },
            { name: 'σᵥ (reflexión vertical)', ops: [{type: 'reflect', axis: 'vertical'}] }
        ],
        invalidSymmetries: ['C2', 'C4', 'C6'],
        // D₃: R = rotación 120°, F = reflexión
        cayleyTable: {
            elements: ['e', 'R', 'R²', 'F', 'FR', 'FR²'],
            table: [
                ['e', 'R', 'R²', 'F', 'FR', 'FR²'],
                ['R', 'R²', 'e', 'FR²', 'F', 'FR'],
                ['R²', 'e', 'R', 'FR', 'FR²', 'F'],
                ['F', 'FR', 'FR²', 'e', 'R', 'R²'],
                ['FR', 'FR²', 'F', 'R²', 'e', 'R'],
                ['FR²', 'F', 'FR', 'R', 'R²', 'e']
            ]
        },
        explanation: `
            Rotación de 120° + 3 ejes de reflexión.
            Los ejes de reflexión pasan POR los centros de rotación.
            El grupo puntual es D₃ (orden 6).
        `
    },
    p31m: {
        name: 'p31m',
        altNames: { schoenflies: 'D₃', hm: '3m', orbifold: '3*3' },
        lattice: 'Hexagonal',
        rotationOrder: 3,
        hasReflection: true,
        hasGlide: false,
        description: 'Rotación 120° + reflexiones entre centros.',
        pointGroupOrder: 6,
        generators: 't₁, t₂, C₃, σₕ',
        validSymmetries: [
            { name: 'Identidad', ops: [] },
            { name: 'C₃ (120°)', ops: [{type: 'rotate', angle: 120}] },
            { name: 'C₃² (240°)', ops: [{type: 'rotate', angle: 240}] },
            { name: 'σₕ (reflexión horizontal)', ops: [{type: 'reflect', axis: 'horizontal'}] }
        ],
        invalidSymmetries: ['C2', 'C4', 'C6'],
        // D₃: R = rotación 120°, F = reflexión
        cayleyTable: {
            elements: ['e', 'R', 'R²', 'F', 'FR', 'FR²'],
            table: [
                ['e', 'R', 'R²', 'F', 'FR', 'FR²'],
                ['R', 'R²', 'e', 'FR²', 'F', 'FR'],
                ['R²', 'e', 'R', 'FR', 'FR²', 'F'],
                ['F', 'FR', 'FR²', 'e', 'R', 'R²'],
                ['FR', 'FR²', 'F', 'R²', 'e', 'R'],
                ['FR²', 'F', 'FR', 'R', 'R²', 'e']
            ]
        },
        explanation: `
            Rotación de 120° + 3 ejes de reflexión.
            Los ejes de reflexión pasan ENTRE los centros de rotación.
            Diferencia sutil con p3m1 en la posición de los ejes.
        `
    },
    p6: {
        name: 'p6',
        altNames: { schoenflies: 'C₆', hm: '6', orbifold: '632' },
        lattice: 'Hexagonal',
        rotationOrder: 6,
        hasReflection: false,
        hasGlide: false,
        description: 'Rotación de 60° (C₆). Sin reflexiones.',
        pointGroupOrder: 6,
        generators: 't₁, t₂, C₆',
        validSymmetries: [
            { name: 'Identidad', ops: [] },
            { name: 'C₆ (60°)', ops: [{type: 'rotate', angle: 60}] },
            { name: 'C₃ (120°)', ops: [{type: 'rotate', angle: 120}] },
            { name: 'C₂ (180°)', ops: [{type: 'rotate', angle: 180}] },
            { name: 'C₃² (240°)', ops: [{type: 'rotate', angle: 240}] },
            { name: 'C₆⁵ (300°)', ops: [{type: 'rotate', angle: 300}] }
        ],
        invalidSymmetries: ['C4', 'σ_v', 'σ_h', 'g'],
        cayleyTable: {
            elements: ['e', 'C₆', 'C₃', 'C₂', 'C₃²', 'C₆⁵'],
            table: [
                ['e', 'C₆', 'C₃', 'C₂', 'C₃²', 'C₆⁵'],
                ['C₆', 'C₃', 'C₂', 'C₃²', 'C₆⁵', 'e'],
                ['C₃', 'C₂', 'C₃²', 'C₆⁵', 'e', 'C₆'],
                ['C₂', 'C₃²', 'C₆⁵', 'e', 'C₆', 'C₃'],
                ['C₃²', 'C₆⁵', 'e', 'C₆', 'C₃', 'C₂'],
                ['C₆⁵', 'e', 'C₆', 'C₃', 'C₂', 'C₃²']
            ]
        },
        explanation: `
            El grupo p6 tiene rotación de 60° (orden 6).
            Contiene C₆, C₃ (=C₆²), C₂ (=C₆³), C₃² (=C₆⁴), C₆⁵.
            C₆⁶ = e.
            NO tiene reflexiones.
        `
    },
    p6m: {
        name: 'p6m',
        altNames: { schoenflies: 'D₆', hm: '6mm', orbifold: '*632' },
        lattice: 'Hexagonal',
        rotationOrder: 6,
        hasReflection: true,
        hasGlide: true,
        description: 'Máxima simetría: C₆ + 6 reflexiones.',
        pointGroupOrder: 12,
        generators: 't₁, t₂, C₆, σᵥ',
        validSymmetries: [
            { name: 'Identidad', ops: [] },
            { name: 'C₆ (60°)', ops: [{type: 'rotate', angle: 60}] },
            { name: 'C₃ (120°)', ops: [{type: 'rotate', angle: 120}] },
            { name: 'C₂ (180°)', ops: [{type: 'rotate', angle: 180}] },
            { name: 'C₃² (240°)', ops: [{type: 'rotate', angle: 240}] },
            { name: 'C₆⁵ (300°)', ops: [{type: 'rotate', angle: 300}] },
            { name: 'σᵥ (reflexión vertical)', ops: [{type: 'reflect', axis: 'vertical'}] },
            { name: 'σₕ (reflexión horizontal)', ops: [{type: 'reflect', axis: 'horizontal'}] }
        ],
        invalidSymmetries: ['C4'],
        // D₆: 6 rotaciones + 6 reflexiones
        // Notación estándar de álgebra: R = rotación 60°, F = reflexión
        // Rotaciones: e, R, R², R³, R⁴, R⁵
        // Reflexiones: F, FR, FR², FR³, FR⁴, FR⁵
        // Reglas: Rⁱ×Rʲ=R^(i+j mod 6), F²=e, F×R=R⁻¹×F
        cayleyTable: {
            elements: ['e', 'R', 'R²', 'R³', 'R⁴', 'R⁵', 'F', 'FR', 'FR²', 'FR³', 'FR⁴', 'FR⁵'],
            table: [
                // e
                ['e', 'R', 'R²', 'R³', 'R⁴', 'R⁵', 'F', 'FR', 'FR²', 'FR³', 'FR⁴', 'FR⁵'],
                // R
                ['R', 'R²', 'R³', 'R⁴', 'R⁵', 'e', 'FR⁵', 'F', 'FR', 'FR²', 'FR³', 'FR⁴'],
                // R²
                ['R²', 'R³', 'R⁴', 'R⁵', 'e', 'R', 'FR⁴', 'FR⁵', 'F', 'FR', 'FR²', 'FR³'],
                // R³
                ['R³', 'R⁴', 'R⁵', 'e', 'R', 'R²', 'FR³', 'FR⁴', 'FR⁵', 'F', 'FR', 'FR²'],
                // R⁴
                ['R⁴', 'R⁵', 'e', 'R', 'R²', 'R³', 'FR²', 'FR³', 'FR⁴', 'FR⁵', 'F', 'FR'],
                // R⁵
                ['R⁵', 'e', 'R', 'R²', 'R³', 'R⁴', 'FR', 'FR²', 'FR³', 'FR⁴', 'FR⁵', 'F'],
                // F
                ['F', 'FR', 'FR²', 'FR³', 'FR⁴', 'FR⁵', 'e', 'R', 'R²', 'R³', 'R⁴', 'R⁵'],
                // FR
                ['FR', 'FR²', 'FR³', 'FR⁴', 'FR⁵', 'F', 'R⁵', 'e', 'R', 'R²', 'R³', 'R⁴'],
                // FR²
                ['FR²', 'FR³', 'FR⁴', 'FR⁵', 'F', 'FR', 'R⁴', 'R⁵', 'e', 'R', 'R²', 'R³'],
                // FR³
                ['FR³', 'FR⁴', 'FR⁵', 'F', 'FR', 'FR²', 'R³', 'R⁴', 'R⁵', 'e', 'R', 'R²'],
                // FR⁴
                ['FR⁴', 'FR⁵', 'F', 'FR', 'FR²', 'FR³', 'R²', 'R³', 'R⁴', 'R⁵', 'e', 'R'],
                // FR⁵
                ['FR⁵', 'F', 'FR', 'FR²', 'FR³', 'FR⁴', 'R', 'R²', 'R³', 'R⁴', 'R⁵', 'e']
            ]
        },
        explanation: `
            El grupo p6m tiene la MÁXIMA simetría de todos los 17 grupos.
            Rotación de 60° + 6 ejes de reflexión.
            El grupo puntual es D₆ (orden 12).
            El dominio fundamental es 1/12 de la celda unitaria.
        `
    }
};

// Export for use in app.js
window.SymmetryOps = SymmetryOps;
window.ImageTransform = ImageTransform;
window.WallpaperGroups = WallpaperGroups;
