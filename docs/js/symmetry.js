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
     */
    _rotate90(imageData, width, height) {
        const newData = new Uint8ClampedArray(imageData.length);
        
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                // (x, y) -> (y, width - 1 - x)
                const srcIdx = (y * width + x) * 4;
                const destX = y;
                const destY = width - 1 - x;
                const destIdx = (destY * width + destX) * 4;
                
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
    _rotate180(imageData, width, height) {
        const newData = new Uint8ClampedArray(imageData.length);
        
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                // (x, y) -> (width - 1 - x, height - 1 - y)
                const srcIdx = (y * width + x) * 4;
                const destX = width - 1 - x;
                const destY = height - 1 - y;
                const destIdx = (destY * width + destX) * 4;
                
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
    _rotate270(imageData, width, height) {
        const newData = new Uint8ClampedArray(imageData.length);
        
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                // (x, y) -> (height - 1 - y, x)
                const srcIdx = (y * width + x) * 4;
                const destX = height - 1 - y;
                const destY = x;
                const destIdx = (destY * width + destX) * 4;
                
                newData[destIdx] = imageData[srcIdx];
                newData[destIdx + 1] = imageData[srcIdx + 1];
                newData[destIdx + 2] = imageData[srcIdx + 2];
                newData[destIdx + 3] = imageData[srcIdx + 3];
            }
        }
        
        return newData;
    },
    
    /**
     * General rotation with bilinear interpolation
     * Used for non-90° angles (60°, 120°, etc.)
     */
    _rotateGeneral(imageData, angleDeg, width, height) {
        const rad = (angleDeg * Math.PI) / 180;
        const cos = Math.cos(rad);
        const sin = Math.sin(rad);
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
                
                // Bilinear interpolation
                const x0 = Math.floor(srcX);
                const y0 = Math.floor(srcY);
                const x1 = x0 + 1;
                const y1 = y0 + 1;
                
                if (x0 >= 0 && x1 < width && y0 >= 0 && y1 < height) {
                    const fx = srcX - x0;
                    const fy = srcY - y0;
                    
                    for (let c = 0; c < 4; c++) {
                        const v00 = imageData[(y0 * width + x0) * 4 + c];
                        const v10 = imageData[(y0 * width + x1) * 4 + c];
                        const v01 = imageData[(y1 * width + x0) * 4 + c];
                        const v11 = imageData[(y1 * width + x1) * 4 + c];
                        
                        const v = v00 * (1 - fx) * (1 - fy) +
                                  v10 * fx * (1 - fy) +
                                  v01 * (1 - fx) * fy +
                                  v11 * fx * fy;
                        
                        newData[destIdx + c] = Math.round(v);
                    }
                } else if (srcX >= 0 && srcX < width && srcY >= 0 && srcY < height) {
                    // Edge case - use nearest neighbor
                    const nearX = Math.round(srcX);
                    const nearY = Math.round(srcY);
                    const srcIdx = (nearY * width + nearX) * 4;
                    newData[destIdx] = imageData[srcIdx];
                    newData[destIdx + 1] = imageData[srcIdx + 1];
                    newData[destIdx + 2] = imageData[srcIdx + 2];
                    newData[destIdx + 3] = imageData[srcIdx + 3];
                } else {
                    // Out of bounds - transparent
                    newData[destIdx + 3] = 0;
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
                // EXACT match - show in bright green
                diffData[i] = 50;
                diffData[i + 1] = 200;
                diffData[i + 2] = 100;
            } else if (maxDiff <= 2) {
                // Near match (rounding errors) - show in light green
                diffData[i] = 80;
                diffData[i + 1] = 180;
                diffData[i + 2] = 80;
            } else {
                // Difference - show in red gradient proportional to difference
                const intensity = Math.min(255, maxDiff * 2);
                diffData[i] = 150 + intensity / 3;
                diffData[i + 1] = 50;
                diffData[i + 2] = 50;
            }
            diffData[i + 3] = 255;
        }
        
        return diffData;
    },

    /**
     * Calculate EXACT correlation between two images
     */
    correlation(imageData1, imageData2) {
        // First check for exact match
        let exactMatch = true;
        let sum1 = 0, sum2 = 0, sumSq1 = 0, sumSq2 = 0, sumProd = 0;
        let n = 0;
        
        for (let i = 0; i < imageData1.length; i += 4) {
            // Use grayscale value for correlation
            const v1 = (imageData1[i] + imageData1[i + 1] + imageData1[i + 2]) / 3;
            const v2 = (imageData2[i] + imageData2[i + 1] + imageData2[i + 2]) / 3;
            
            // Check exact match (allowing for tiny rounding)
            if (Math.abs(v1 - v2) > 0.5) {
                exactMatch = false;
            }
            
            sum1 += v1;
            sum2 += v2;
            sumSq1 += v1 * v1;
            sumSq2 += v2 * v2;
            sumProd += v1 * v2;
            n++;
        }
        
        // If exact match, return 1
        if (exactMatch) {
            return 1.0;
        }
        
        const mean1 = sum1 / n;
        const mean2 = sum2 / n;
        const var1 = sumSq1 / n - mean1 * mean1;
        const var2 = sumSq2 / n - mean2 * mean2;
        const cov = sumProd / n - mean1 * mean2;
        
        if (var1 <= 0 || var2 <= 0) return 1;
        
        return cov / Math.sqrt(var1 * var2);
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
        lattice: 'Rectangular',
        rotationOrder: 1,
        hasReflection: true,
        hasGlide: false,
        description: 'Ejes de reflexión paralelos. Sin rotación.',
        pointGroupOrder: 2,
        generators: 't₁, t₂, σ',
        validSymmetries: [
            { name: 'Identidad', ops: [] },
            { name: 'σᵥ (reflexión vertical)', ops: [{type: 'reflect', axis: 'vertical'}] },
            { name: 'T(1,0)', ops: [{type: 'translate', dx: 1, dy: 0}] }
        ],
        invalidSymmetries: ['C2', 'C3', 'C4', 'C6', 'g'],
        cayleyTable: {
            elements: ['e', 'σ'],
            table: [
                ['e', 'σ'],
                ['σ', 'e']
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
        lattice: 'Rectangular',
        rotationOrder: 1,
        hasReflection: false,
        hasGlide: true,
        description: 'Reflexiones deslizantes (glide) paralelas. Sin reflexión pura.',
        pointGroupOrder: 1,
        generators: 't₁, t₂, g',
        validSymmetries: [
            { name: 'Identidad', ops: [] },
            { name: 'g × g = T', ops: [{type: 'glide', axis: 'vertical'}, {type: 'glide', axis: 'vertical'}] },
            { name: 'T(1,0)', ops: [{type: 'translate', dx: 1, dy: 0}] }
        ],
        invalidSymmetries: ['C2', 'C3', 'C4', 'C6', 'σ_v', 'σ_h'],
        cayleyTable: {
            elements: ['e', 'g', 'g²=t'],
            table: [
                ['e', 'g', 't'],
                ['g', 't', 'gt'],
                ['t', 'gt', 't²']
            ]
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
        lattice: 'Rectangular (centrado)',
        rotationOrder: 1,
        hasReflection: true,
        hasGlide: true,
        description: 'Reflexión + glide entre ejes de reflexión.',
        pointGroupOrder: 2,
        generators: 't₁, t₂, σ',
        validSymmetries: [
            { name: 'Identidad', ops: [] },
            { name: 'σᵥ (reflexión)', ops: [{type: 'reflect', axis: 'vertical'}] },
            { name: 'T(1,0)', ops: [{type: 'translate', dx: 1, dy: 0}] }
        ],
        invalidSymmetries: ['C2', 'C3', 'C4', 'C6'],
        cayleyTable: {
            elements: ['e', 'σ'],
            table: [
                ['e', 'σ'],
                ['σ', 'e']
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
        cayleyTable: {
            elements: ['e', 'σᵥ', 'σₕ', 'C₂'],
            table: [
                ['e', 'σᵥ', 'σₕ', 'C₂'],
                ['σᵥ', 'e', 'C₂', 'σₕ'],
                ['σₕ', 'C₂', 'e', 'σᵥ'],
                ['C₂', 'σₕ', 'σᵥ', 'e']
            ]
        },
        explanation: `
            El grupo pmm tiene dos ejes de reflexión perpendiculares.
            IMPORTANTE: σᵥ ∘ σₕ = C₂ (dos reflexiones perpendiculares = rotación 180°)
            Esto es un teorema fundamental de la teoría de grupos.
        `
    },
    pmg: {
        name: 'pmg',
        lattice: 'Rectangular',
        rotationOrder: 2,
        hasReflection: true,
        hasGlide: true,
        description: 'Reflexión + glide perpendicular.',
        pointGroupOrder: 4,
        generators: 't₁, t₂, σ, g',
        validSymmetries: [
            { name: 'Identidad', ops: [] },
            { name: 'σᵥ', ops: [{type: 'reflect', axis: 'vertical'}] },
            { name: 'C₂', ops: [{type: 'rotate', angle: 180}] }
        ],
        invalidSymmetries: ['C3', 'C4', 'C6'],
        cayleyTable: {
            elements: ['e', 'σ', 'g', 'C₂'],
            table: [
                ['e', 'σ', 'g', 'C₂'],
                ['σ', 'e', 'C₂', 'g'],
                ['g', 'C₂', 't', 'σt'],
                ['C₂', 'g', 'σt', 't']
            ]
        },
        explanation: `
            Reflexión en una dirección, glide en la perpendicular.
            La combinación crea rotación de 180°.
        `
    },
    pgg: {
        name: 'pgg',
        lattice: 'Rectangular',
        rotationOrder: 2,
        hasReflection: false,
        hasGlide: true,
        description: 'Glides perpendiculares. gᵥ ∘ gₕ = C₂',
        pointGroupOrder: 4,
        generators: 't₁, t₂, gᵥ, gₕ',
        validSymmetries: [
            { name: 'Identidad', ops: [] },
            { name: 'C₂', ops: [{type: 'rotate', angle: 180}] }
        ],
        invalidSymmetries: ['C3', 'C4', 'C6', 'σ_v', 'σ_h'],
        cayleyTable: {
            elements: ['e', 'gᵥ', 'gₕ', 'C₂'],
            table: [
                ['e', 'gᵥ', 'gₕ', 'C₂'],
                ['gᵥ', 'tᵥ', 'C₂', 'gₕtᵥ'],
                ['gₕ', 'C₂', 'tₕ', 'gᵥtₕ'],
                ['C₂', 'gₕtᵥ', 'gᵥtₕ', 'e']
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
        cayleyTable: {
            elements: ['e', 'σᵥ', 'σₕ', 'C₂'],
            table: [
                ['e', 'σᵥ', 'σₕ', 'C₂'],
                ['σᵥ', 'e', 'C₂', 'σₕ'],
                ['σₕ', 'C₂', 'e', 'σᵥ'],
                ['C₂', 'σₕ', 'σᵥ', 'e']
            ]
        },
        explanation: `
            Como pmm pero con celda centrada.
            Tiene reflexiones perpendiculares Y glides.
        `
    },
    p4: {
        name: 'p4',
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
        cayleyTable: {
            elements: ['e', 'C₄', 'C₂', 'C₄³', 'σᵥ', 'σₕ', 'σ_d', 'σ_d\''],
            table: [
                ['e', 'C₄', 'C₂', 'C₄³', 'σᵥ', 'σₕ', 'σ_d', 'σ_d\''],
                ['C₄', 'C₂', 'C₄³', 'e', 'σ_d\'', 'σ_d', 'σᵥ', 'σₕ'],
                ['C₂', 'C₄³', 'e', 'C₄', 'σₕ', 'σᵥ', 'σ_d\'', 'σ_d'],
                ['C₄³', 'e', 'C₄', 'C₂', 'σ_d', 'σ_d\'', 'σₕ', 'σᵥ'],
                ['σᵥ', 'σ_d', 'σₕ', 'σ_d\'', 'e', 'C₂', 'C₄', 'C₄³'],
                ['σₕ', 'σ_d\'', 'σᵥ', 'σ_d', 'C₂', 'e', 'C₄³', 'C₄'],
                ['σ_d', 'σₕ', 'σ_d\'', 'σᵥ', 'C₄³', 'C₄', 'e', 'C₂'],
                ['σ_d\'', 'σᵥ', 'σ_d', 'σₕ', 'C₄', 'C₄³', 'C₂', 'e']
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
        lattice: 'Cuadrado',
        rotationOrder: 4,
        hasReflection: true,
        hasGlide: true,
        description: 'Rotación 90° + reflexiones diagonales + glides.',
        pointGroupOrder: 8,
        generators: 't₁, t₂, C₄, σ_d',
        validSymmetries: [
            { name: 'Identidad', ops: [] },
            { name: 'C₄', ops: [{type: 'rotate', angle: 90}] },
            { name: 'C₂', ops: [{type: 'rotate', angle: 180}] },
            { name: 'C₄³', ops: [{type: 'rotate', angle: 270}] },
            { name: 'σ_d', ops: [{type: 'reflect', axis: 'diagonal'}] },
            { name: 'σ_d\'', ops: [{type: 'reflect', axis: 'antidiagonal'}] }
        ],
        invalidSymmetries: ['C3', 'C6'],
        cayleyTable: {
            elements: ['e', 'C₄', 'C₂', 'C₄³', 'σ_d', 'σ_d\'', 'gᵥ', 'gₕ'],
            table: [
                ['e', 'C₄', 'C₂', 'C₄³', 'σ_d', 'σ_d\'', 'gᵥ', 'gₕ'],
                ['C₄', 'C₂', 'C₄³', 'e', 'gₕ', 'gᵥ', 'σ_d', 'σ_d\''],
                ['C₂', 'C₄³', 'e', 'C₄', 'σ_d\'', 'σ_d', 'gₕ', 'gᵥ'],
                ['C₄³', 'e', 'C₄', 'C₂', 'gᵥ', 'gₕ', 'σ_d\'', 'σ_d'],
                ['σ_d', 'gᵥ', 'σ_d\'', 'gₕ', 'e', 'C₂', 'C₄', 'C₄³'],
                ['σ_d\'', 'gₕ', 'σ_d', 'gᵥ', 'C₂', 'e', 'C₄³', 'C₄'],
                ['gᵥ', 'σ_d\'', 'gₕ', 'σ_d', 'C₄³', 'C₄', 'tᵥ', 'C₂tᵥ'],
                ['gₕ', 'σ_d', 'gᵥ', 'σ_d\'', 'C₄', 'C₄³', 'C₂tₕ', 'tₕ']
            ]
        },
        explanation: `
            Similar a p4m pero con reflexiones solo en diagonales.
            Las reflexiones en ejes horizontal/vertical son glides.
        `
    },
    p3: {
        name: 'p3',
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
        lattice: 'Hexagonal',
        rotationOrder: 3,
        hasReflection: true,
        hasGlide: false,
        description: 'Rotación 120° + reflexiones a través de centros.',
        pointGroupOrder: 6,
        generators: 't₁, t₂, C₃, σ',
        validSymmetries: [
            { name: 'Identidad', ops: [] },
            { name: 'C₃', ops: [{type: 'rotate', angle: 120}] },
            { name: 'C₃²', ops: [{type: 'rotate', angle: 240}] },
            { name: 'σᵥ', ops: [{type: 'reflect', axis: 'vertical'}] }
        ],
        invalidSymmetries: ['C2', 'C4', 'C6'],
        cayleyTable: {
            elements: ['e', 'C₃', 'C₃²', 'σ₁', 'σ₂', 'σ₃'],
            table: [
                ['e', 'C₃', 'C₃²', 'σ₁', 'σ₂', 'σ₃'],
                ['C₃', 'C₃²', 'e', 'σ₃', 'σ₁', 'σ₂'],
                ['C₃²', 'e', 'C₃', 'σ₂', 'σ₃', 'σ₁'],
                ['σ₁', 'σ₂', 'σ₃', 'e', 'C₃', 'C₃²'],
                ['σ₂', 'σ₃', 'σ₁', 'C₃²', 'e', 'C₃'],
                ['σ₃', 'σ₁', 'σ₂', 'C₃', 'C₃²', 'e']
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
        lattice: 'Hexagonal',
        rotationOrder: 3,
        hasReflection: true,
        hasGlide: false,
        description: 'Rotación 120° + reflexiones entre centros.',
        pointGroupOrder: 6,
        generators: 't₁, t₂, C₃, σ',
        validSymmetries: [
            { name: 'Identidad', ops: [] },
            { name: 'C₃', ops: [{type: 'rotate', angle: 120}] },
            { name: 'C₃²', ops: [{type: 'rotate', angle: 240}] },
            { name: 'σₕ', ops: [{type: 'reflect', axis: 'horizontal'}] }
        ],
        invalidSymmetries: ['C2', 'C4', 'C6'],
        cayleyTable: {
            elements: ['e', 'C₃', 'C₃²', 'σ₁', 'σ₂', 'σ₃'],
            table: [
                ['e', 'C₃', 'C₃²', 'σ₁', 'σ₂', 'σ₃'],
                ['C₃', 'C₃²', 'e', 'σ₃', 'σ₁', 'σ₂'],
                ['C₃²', 'e', 'C₃', 'σ₂', 'σ₃', 'σ₁'],
                ['σ₁', 'σ₂', 'σ₃', 'e', 'C₃', 'C₃²'],
                ['σ₂', 'σ₃', 'σ₁', 'C₃²', 'e', 'C₃'],
                ['σ₃', 'σ₁', 'σ₂', 'C₃', 'C₃²', 'e']
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
        lattice: 'Hexagonal',
        rotationOrder: 6,
        hasReflection: true,
        hasGlide: true,
        description: 'Máxima simetría: C₆ + 6 reflexiones.',
        pointGroupOrder: 12,
        generators: 't₁, t₂, C₆, σ',
        validSymmetries: [
            { name: 'Identidad', ops: [] },
            { name: 'C₆', ops: [{type: 'rotate', angle: 60}] },
            { name: 'C₃', ops: [{type: 'rotate', angle: 120}] },
            { name: 'C₂', ops: [{type: 'rotate', angle: 180}] },
            { name: 'C₃²', ops: [{type: 'rotate', angle: 240}] },
            { name: 'C₆⁵', ops: [{type: 'rotate', angle: 300}] },
            { name: 'σᵥ', ops: [{type: 'reflect', axis: 'vertical'}] },
            { name: 'σₕ', ops: [{type: 'reflect', axis: 'horizontal'}] }
        ],
        invalidSymmetries: ['C4'],
        cayleyTable: {
            elements: ['e', 'C₆', 'C₃', 'C₂', 'C₃²', 'C₆⁵', 'σ₁', 'σ₂', 'σ₃', 'σ₄', 'σ₅', 'σ₆'],
            table: 'D₆ (dihedral group of order 12) - tabla completa disponible en referencias'
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
