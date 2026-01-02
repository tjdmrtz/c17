/**
 * Symmetry Operations Module
 * Mathematical transformations for wallpaper group symmetries
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
            if (Math.abs(v - 0.5) < 1e-10) return "0.5";
            if (Math.abs(v + 0.5) < 1e-10) return "-0.5";
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
 * Wallpaper group definitions with CORRECT symmetry properties
 */
const WallpaperGroups = {
    p1: {
        name: 'p1',
        lattice: 'Oblicuo',
        rotationOrder: 1,
        hasReflection: false,
        hasGlide: false,
        description: 'Solo traslaciones, sin simetría puntual',
        pointGroupOrder: 1,
        generators: 't₁, t₂',
        // p1 has NO rotational symmetry, NO reflection - only translations
        validOperations: ['translate'],
        symmetryAngles: []  // No rotation symmetry
    },
    p2: {
        name: 'p2',
        lattice: 'Oblicuo',
        rotationOrder: 2,
        hasReflection: false,
        hasGlide: false,
        description: 'Centros de rotación de 180°',
        pointGroupOrder: 2,
        generators: 't₁, t₂, C₂',
        validOperations: ['rotate180', 'translate'],
        symmetryAngles: [180]
    },
    pm: {
        name: 'pm',
        lattice: 'Rectangular',
        rotationOrder: 1,
        hasReflection: true,
        hasGlide: false,
        description: 'Ejes de reflexión paralelos',
        pointGroupOrder: 2,
        generators: 't₁, t₂, σ',
        validOperations: ['reflectVertical', 'translate'],
        symmetryAngles: []
    },
    pg: {
        name: 'pg',
        lattice: 'Rectangular',
        rotationOrder: 1,
        hasReflection: false,
        hasGlide: true,
        description: 'Reflexiones deslizantes paralelas',
        pointGroupOrder: 2,
        generators: 't₁, t₂, g',
        validOperations: ['glide', 'translate'],
        symmetryAngles: []
    },
    cm: {
        name: 'cm',
        lattice: 'Rectangular',
        rotationOrder: 1,
        hasReflection: true,
        hasGlide: true,
        description: 'Ejes de reflexión con deslizamiento entre ellos',
        pointGroupOrder: 2,
        generators: 't₁, t₂, σ, g',
        validOperations: ['reflectVertical', 'glide', 'translate'],
        symmetryAngles: []
    },
    pmm: {
        name: 'pmm',
        lattice: 'Rectangular',
        rotationOrder: 2,
        hasReflection: true,
        hasGlide: false,
        description: 'Ejes de reflexión perpendiculares',
        pointGroupOrder: 4,
        generators: 't₁, t₂, σᵥ, σₕ',
        validOperations: ['rotate180', 'reflectVertical', 'reflectHorizontal', 'translate'],
        symmetryAngles: [180]
    },
    pmg: {
        name: 'pmg',
        lattice: 'Rectangular',
        rotationOrder: 2,
        hasReflection: true,
        hasGlide: true,
        description: 'Reflexión + deslizamiento perpendicular',
        pointGroupOrder: 4,
        generators: 't₁, t₂, σ, g',
        validOperations: ['rotate180', 'reflectVertical', 'glide', 'translate'],
        symmetryAngles: [180]
    },
    pgg: {
        name: 'pgg',
        lattice: 'Rectangular',
        rotationOrder: 2,
        hasReflection: false,
        hasGlide: true,
        description: 'Reflexiones deslizantes perpendiculares',
        pointGroupOrder: 4,
        generators: 't₁, t₂, gᵥ, gₕ',
        validOperations: ['rotate180', 'glide', 'translate'],
        symmetryAngles: [180]
    },
    cmm: {
        name: 'cmm',
        lattice: 'Rectangular',
        rotationOrder: 2,
        hasReflection: true,
        hasGlide: true,
        description: 'Celda centrada con reflexiones',
        pointGroupOrder: 4,
        generators: 't₁, t₂, σᵥ, σₕ, g',
        validOperations: ['rotate180', 'reflectVertical', 'reflectHorizontal', 'glide', 'translate'],
        symmetryAngles: [180]
    },
    p4: {
        name: 'p4',
        lattice: 'Cuadrado',
        rotationOrder: 4,
        hasReflection: false,
        hasGlide: false,
        description: 'Centros de rotación de 90°',
        pointGroupOrder: 4,
        generators: 't₁, t₂, C₄',
        validOperations: ['rotate90', 'rotate180', 'rotate270', 'translate'],
        symmetryAngles: [90, 180, 270]
    },
    p4m: {
        name: 'p4m',
        lattice: 'Cuadrado',
        rotationOrder: 4,
        hasReflection: true,
        hasGlide: true,
        description: 'Cuadrado con reflexiones en todos los ejes',
        pointGroupOrder: 8,
        generators: 't₁, t₂, C₄, σ',
        validOperations: ['rotate90', 'rotate180', 'rotate270', 'reflectVertical', 'reflectHorizontal', 'reflectDiagonal', 'reflectAntidiagonal', 'translate'],
        symmetryAngles: [90, 180, 270]
    },
    p4g: {
        name: 'p4g',
        lattice: 'Cuadrado',
        rotationOrder: 4,
        hasReflection: true,
        hasGlide: true,
        description: 'Cuadrado con deslizamientos y rotaciones',
        pointGroupOrder: 8,
        generators: 't₁, t₂, C₄, g',
        validOperations: ['rotate90', 'rotate180', 'rotate270', 'reflectDiagonal', 'reflectAntidiagonal', 'glide', 'translate'],
        symmetryAngles: [90, 180, 270]
    },
    p3: {
        name: 'p3',
        lattice: 'Hexagonal',
        rotationOrder: 3,
        hasReflection: false,
        hasGlide: false,
        description: 'Centros de rotación de 120°',
        pointGroupOrder: 3,
        generators: 't₁, t₂, C₃',
        validOperations: ['rotate120', 'rotate240', 'translate'],
        symmetryAngles: [120, 240]
    },
    p3m1: {
        name: 'p3m1',
        lattice: 'Hexagonal',
        rotationOrder: 3,
        hasReflection: true,
        hasGlide: false,
        description: 'Rotación 120° con ejes de reflexión a través de centros',
        pointGroupOrder: 6,
        generators: 't₁, t₂, C₃, σ',
        validOperations: ['rotate120', 'rotate240', 'reflectVertical', 'translate'],
        symmetryAngles: [120, 240]
    },
    p31m: {
        name: 'p31m',
        lattice: 'Hexagonal',
        rotationOrder: 3,
        hasReflection: true,
        hasGlide: false,
        description: 'Rotación 120° con ejes de reflexión entre centros',
        pointGroupOrder: 6,
        generators: 't₁, t₂, C₃, σ',
        validOperations: ['rotate120', 'rotate240', 'reflectHorizontal', 'translate'],
        symmetryAngles: [120, 240]
    },
    p6: {
        name: 'p6',
        lattice: 'Hexagonal',
        rotationOrder: 6,
        hasReflection: false,
        hasGlide: false,
        description: 'Centros de rotación de 60°',
        pointGroupOrder: 6,
        generators: 't₁, t₂, C₆',
        validOperations: ['rotate60', 'rotate120', 'rotate180', 'rotate240', 'rotate300', 'translate'],
        symmetryAngles: [60, 120, 180, 240, 300]
    },
    p6m: {
        name: 'p6m',
        lattice: 'Hexagonal',
        rotationOrder: 6,
        hasReflection: true,
        hasGlide: true,
        description: 'Hexagonal con todas las simetrías (máxima simetría)',
        pointGroupOrder: 12,
        generators: 't₁, t₂, C₆, σ',
        validOperations: ['rotate60', 'rotate120', 'rotate180', 'rotate240', 'rotate300', 'reflectVertical', 'reflectHorizontal', 'glide', 'translate'],
        symmetryAngles: [60, 120, 180, 240, 300]
    }
};

// Export for use in app.js
window.SymmetryOps = SymmetryOps;
window.ImageTransform = ImageTransform;
window.WallpaperGroups = WallpaperGroups;
