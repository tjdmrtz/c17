/**
 * Generador de patrones de wallpaper usando la técnica del dominio fundamental.
 * 
 * Basado en el FixedWallpaperGenerator de Python que genera patrones
 * con simetrías matemáticamente correctas.
 * 
 * Técnica: Crear un motivo asimétrico y aplicar las operaciones del grupo
 * para construir la celda unitaria, luego repetir (tile).
 */

const PatternGenerator = {
    
    /**
     * Crear motivo asimétrico aleatorio usando Gaussianas
     */
    createMotif(size, rng, complexity = 3) {
        const motif = new Float32Array(size * size);
        
        for (let k = 0; k < complexity; k++) {
            const cx = rng() * size;
            const cy = rng() * size;
            const sigma = rng() * size / 4 + size / 10;
            const amplitude = rng() * 0.5 + 0.5;
            
            for (let y = 0; y < size; y++) {
                for (let x = 0; x < size; x++) {
                    const dx = x - cx;
                    const dy = y - cy;
                    const value = amplitude * Math.exp(-(dx*dx + dy*dy) / (2 * sigma * sigma));
                    motif[y * size + x] += value;
                }
            }
        }
        
        // Normalizar a [0, 1]
        let max = 0;
        for (let i = 0; i < motif.length; i++) {
            if (motif[i] > max) max = motif[i];
        }
        if (max > 0) {
            for (let i = 0; i < motif.length; i++) {
                motif[i] /= max;
            }
        }
        
        return motif;
    },
    
    /**
     * Flip horizontal (left-right)
     */
    flipLR(data, width, height) {
        const result = new Float32Array(width * height);
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                result[y * width + x] = data[y * width + (width - 1 - x)];
            }
        }
        return result;
    },
    
    /**
     * Flip vertical (up-down)
     */
    flipUD(data, width, height) {
        const result = new Float32Array(width * height);
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                result[y * width + x] = data[(height - 1 - y) * width + x];
            }
        }
        return result;
    },
    
    /**
     * Rotar 90° en sentido antihorario
     */
    rot90(data, size) {
        const result = new Float32Array(size * size);
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                // (x, y) -> (y, size-1-x)
                result[x * size + (size - 1 - y)] = data[y * size + x];
            }
        }
        return result;
    },
    
    /**
     * Rotar 180°
     */
    rot180(data, size) {
        const result = new Float32Array(size * size);
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                result[(size - 1 - y) * size + (size - 1 - x)] = data[y * size + x];
            }
        }
        return result;
    },
    
    /**
     * Rotar 270° (o -90°)
     */
    rot270(data, size) {
        const result = new Float32Array(size * size);
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                // (x, y) -> (size-1-y, x)
                result[(size - 1 - x) * size + y] = data[y * size + x];
            }
        }
        return result;
    },
    
    /**
     * Combinar horizontalmente dos arrays [A | B]
     */
    hstack(a, b, width, height) {
        const result = new Float32Array(height * width * 2);
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                result[y * width * 2 + x] = a[y * width + x];
                result[y * width * 2 + width + x] = b[y * width + x];
            }
        }
        return { data: result, width: width * 2, height: height };
    },
    
    /**
     * Combinar verticalmente dos arrays [A; B]
     */
    vstack(a, b, width, height) {
        const result = new Float32Array(height * 2 * width);
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                result[y * width + x] = a[y * width + x];
                result[(height + y) * width + x] = b[y * width + x];
            }
        }
        return { data: result, width: width, height: height * 2 };
    },
    
    /**
     * Colocar un bloque en una posición de una matriz más grande
     */
    placeBlock(target, targetWidth, block, blockWidth, blockHeight, offsetX, offsetY) {
        for (let y = 0; y < blockHeight; y++) {
            for (let x = 0; x < blockWidth; x++) {
                const ty = offsetY + y;
                const tx = offsetX + x;
                target[ty * targetWidth + tx] = block[y * blockWidth + x];
            }
        }
    },
    
    /**
     * Tile (repetir) la celda para llenar el patrón
     */
    tile(cell, cellWidth, cellHeight, targetSize) {
        const result = new Float32Array(targetSize * targetSize);
        
        for (let y = 0; y < targetSize; y++) {
            for (let x = 0; x < targetSize; x++) {
                const cy = y % cellHeight;
                const cx = x % cellWidth;
                result[y * targetSize + x] = cell[cy * cellWidth + cx];
            }
        }
        
        return result;
    },
    
    // ===== GENERADORES POR GRUPO =====
    
    /**
     * p1: Solo traslaciones
     */
    generateP1(size, rng, motifSize = 64) {
        const motif = this.createMotif(motifSize, rng);
        return this.tile(motif, motifSize, motifSize, size);
    },
    
    /**
     * p2: Rotación 180° 
     * Celda: [fund | ] + rot180 en esquina opuesta
     */
    generateP2(size, rng, motifSize = 64) {
        const halfSize = motifSize;
        const cellSize = halfSize * 2;
        
        const fund = this.createMotif(halfSize, rng);
        const cell = new Float32Array(cellSize * cellSize);
        
        // Upper-left = fund
        this.placeBlock(cell, cellSize, fund, halfSize, halfSize, 0, 0);
        // Lower-right = rot180(fund)
        const rotated = this.rot180(fund, halfSize);
        this.placeBlock(cell, cellSize, rotated, halfSize, halfSize, halfSize, halfSize);
        
        return this.tile(cell, cellSize, cellSize, size);
    },
    
    /**
     * pm: Reflexión vertical
     * Celda: [motif | flipLR(motif)]
     */
    generatePM(size, rng, motifSize = 64) {
        const motif = this.createMotif(motifSize, rng);
        const flipped = this.flipLR(motif, motifSize, motifSize);
        const { data, width, height } = this.hstack(motif, flipped, motifSize, motifSize);
        return this.tile(data, width, height, size);
    },
    
    /**
     * pg: Glide reflection
     */
    generatePG(size, rng, motifSize = 64) {
        const motif = this.createMotif(motifSize, rng);
        
        // Glide = flipUD + roll horizontal
        const flipped = this.flipUD(motif, motifSize, motifSize);
        const glided = new Float32Array(motifSize * motifSize);
        const shift = Math.floor(motifSize / 2);
        for (let y = 0; y < motifSize; y++) {
            for (let x = 0; x < motifSize; x++) {
                const srcX = (x - shift + motifSize) % motifSize;
                glided[y * motifSize + x] = flipped[y * motifSize + srcX];
            }
        }
        
        const { data, width, height } = this.hstack(motif, glided, motifSize, motifSize);
        return this.tile(data, width, height, size);
    },
    
    /**
     * cm: Centered reflection
     */
    generateCM(size, rng, motifSize = 64) {
        const motif = this.createMotif(motifSize, rng);
        const flipped = this.flipLR(motif, motifSize, motifSize);
        
        // Row 1: [motif | flipLR(motif)]
        const row1 = this.hstack(motif, flipped, motifSize, motifSize);
        
        // Row 2: shifted by half period
        const row2Data = new Float32Array(row1.data.length);
        for (let y = 0; y < row1.height; y++) {
            for (let x = 0; x < row1.width; x++) {
                const srcX = (x - motifSize + row1.width) % row1.width;
                row2Data[y * row1.width + x] = row1.data[y * row1.width + srcX];
            }
        }
        
        const cell = this.vstack(row1.data, row2Data, row1.width, row1.height);
        return this.tile(cell.data, cell.width, cell.height, size);
    },
    
    /**
     * pmm: Reflexiones perpendiculares (D2)
     * | M         | flipLR(M) |
     * | flipUD(M) | rot180(M) |
     */
    generatePMM(size, rng, motifSize = 64) {
        const motif = this.createMotif(motifSize, rng);
        
        const flipH = this.flipLR(motif, motifSize, motifSize);
        const flipV = this.flipUD(motif, motifSize, motifSize);
        const rot = this.rot180(motif, motifSize);
        
        const top = this.hstack(motif, flipH, motifSize, motifSize);
        const bottom = this.hstack(flipV, rot, motifSize, motifSize);
        const cell = this.vstack(top.data, bottom.data, top.width, top.height);
        
        return this.tile(cell.data, cell.width, cell.height, size);
    },
    
    /**
     * pmg: Reflexión + glide perpendicular
     */
    generatePMG(size, rng, motifSize = 64) {
        const halfSize = motifSize;
        const cellSize = halfSize * 2;
        
        const fund = this.createMotif(halfSize, rng);
        const cell = new Float32Array(cellSize * cellSize);
        
        // Upper left: fund
        this.placeBlock(cell, cellSize, fund, halfSize, halfSize, 0, 0);
        // Upper right: flipLR(fund)
        const flipH = this.flipLR(fund, halfSize, halfSize);
        this.placeBlock(cell, cellSize, flipH, halfSize, halfSize, halfSize, 0);
        // Lower half: rot180 of upper half (ensures C2)
        const upperHalf = new Float32Array(halfSize * cellSize);
        for (let y = 0; y < halfSize; y++) {
            for (let x = 0; x < cellSize; x++) {
                upperHalf[y * cellSize + x] = cell[y * cellSize + x];
            }
        }
        const lowerHalf = this.rot180(upperHalf, cellSize);
        for (let y = 0; y < halfSize; y++) {
            for (let x = 0; x < cellSize; x++) {
                cell[(halfSize + y) * cellSize + x] = lowerHalf[y * cellSize + x];
            }
        }
        
        return this.tile(cell, cellSize, cellSize, size);
    },
    
    /**
     * pgg: Dos glides perpendiculares -> C2
     */
    generatePGG(size, rng, motifSize = 64) {
        const halfSize = motifSize;
        const cellSize = halfSize * 2;
        
        const fund = this.createMotif(halfSize, rng);
        const cell = new Float32Array(cellSize * cellSize);
        
        // Upper left
        this.placeBlock(cell, cellSize, fund, halfSize, halfSize, 0, 0);
        // Lower right: rot180
        const rot = this.rot180(fund, halfSize);
        this.placeBlock(cell, cellSize, rot, halfSize, halfSize, halfSize, halfSize);
        
        // Upper right: glide of fund
        const flipped = this.flipUD(fund, halfSize, halfSize);
        const shift = Math.floor(halfSize / 2);
        const glided = new Float32Array(halfSize * halfSize);
        for (let y = 0; y < halfSize; y++) {
            for (let x = 0; x < halfSize; x++) {
                const srcX = (x - shift + halfSize) % halfSize;
                glided[y * halfSize + x] = flipped[y * halfSize + srcX];
            }
        }
        this.placeBlock(cell, cellSize, glided, halfSize, halfSize, halfSize, 0);
        
        // Lower left: rot180 of upper right
        const glidedRot = this.rot180(glided, halfSize);
        this.placeBlock(cell, cellSize, glidedRot, halfSize, halfSize, 0, halfSize);
        
        return this.tile(cell, cellSize, cellSize, size);
    },
    
    /**
     * cmm: Celda centrada con reflexiones
     */
    generateCMM(size, rng, motifSize = 64) {
        const motif = this.createMotif(motifSize, rng);
        
        const flipH = this.flipLR(motif, motifSize, motifSize);
        const flipV = this.flipUD(motif, motifSize, motifSize);
        const rot = this.rot180(motif, motifSize);
        
        const top = this.hstack(motif, flipH, motifSize, motifSize);
        const bottom = this.hstack(flipV, rot, motifSize, motifSize);
        const basic = this.vstack(top.data, bottom.data, top.width, top.height);
        
        // Add centering
        const cellW = basic.width;
        const cellH = basic.height;
        const shifted = new Float32Array(basic.data.length);
        for (let y = 0; y < cellH; y++) {
            for (let x = 0; x < cellW; x++) {
                const sy = (y + Math.floor(cellH / 2)) % cellH;
                const sx = (x + Math.floor(cellW / 2)) % cellW;
                shifted[y * cellW + x] = basic.data[sy * cellW + sx];
            }
        }
        
        // Combine
        const cell = new Float32Array(basic.data.length);
        for (let i = 0; i < cell.length; i++) {
            cell[i] = basic.data[i] + shifted[i] * 0.5;
        }
        
        // Normalize
        let max = 0;
        for (let i = 0; i < cell.length; i++) {
            if (cell[i] > max) max = cell[i];
        }
        if (max > 0) {
            for (let i = 0; i < cell.length; i++) {
                cell[i] /= max;
            }
        }
        
        return this.tile(cell, cellW, cellH, size);
    },
    
    /**
     * p4: Rotación 90° (C4, sin reflexión)
     */
    generateP4(size, rng, motifSize = 64) {
        const fund = this.createMotif(motifSize, rng);
        const cellSize = motifSize * 2;
        const cell = new Float32Array(cellSize * cellSize);
        
        // Place fund and its rotations in 4 quadrants
        this.placeBlock(cell, cellSize, fund, motifSize, motifSize, 0, 0);
        
        const rot270 = this.rot270(fund, motifSize);
        this.placeBlock(cell, cellSize, rot270, motifSize, motifSize, motifSize, 0);
        
        const rot180 = this.rot180(fund, motifSize);
        this.placeBlock(cell, cellSize, rot180, motifSize, motifSize, motifSize, motifSize);
        
        const rot90 = this.rot90(fund, motifSize);
        this.placeBlock(cell, cellSize, rot90, motifSize, motifSize, 0, motifSize);
        
        return this.tile(cell, cellSize, cellSize, size);
    },
    
    /**
     * p4m: C4 + reflexiones (D4)
     */
    generateP4M(size, rng, motifSize = 64) {
        // Crear motivo con simetría D4 incorporada
        const motif = this.createMotif(motifSize, rng);
        
        // Simetrizar el motivo para D4
        const symMotif = new Float32Array(motifSize * motifSize);
        for (let y = 0; y < motifSize; y++) {
            for (let x = 0; x < motifSize; x++) {
                const val = motif[y * motifSize + x];
                // Aplicar todas las simetrías de D4
                const mx = motifSize - 1 - x;
                const my = motifSize - 1 - y;
                
                symMotif[y * motifSize + x] += val;
                symMotif[y * motifSize + mx] += val;  // flip horizontal
                symMotif[my * motifSize + x] += val;  // flip vertical
                symMotif[my * motifSize + mx] += val; // rot180
                symMotif[x * motifSize + y] += val;   // diagonal reflection
                symMotif[mx * motifSize + my] += val; // anti-diagonal reflection
                symMotif[x * motifSize + my] += val;  // rot90 + flip
                symMotif[mx * motifSize + y] += val;  // rot270 + flip
            }
        }
        
        // Normalizar
        let max = 0;
        for (let i = 0; i < symMotif.length; i++) {
            if (symMotif[i] > max) max = symMotif[i];
        }
        if (max > 0) {
            for (let i = 0; i < symMotif.length; i++) {
                symMotif[i] /= max;
            }
        }
        
        return this.tile(symMotif, motifSize, motifSize, size);
    },
    
    /**
     * p4g: C4 + glide reflections
     */
    generateP4G(size, rng, motifSize = 64) {
        const quarter = this.createMotif(motifSize, rng);
        const cellSize = motifSize * 2;
        const cell = new Float32Array(cellSize * cellSize);
        
        // Place quarter in top-left
        const temp = new Float32Array(cellSize * cellSize);
        this.placeBlock(temp, cellSize, quarter, motifSize, motifSize, 0, 0);
        
        // Accumulate 4 rotations for C4
        for (let k = 0; k < 4; k++) {
            let rotated = temp;
            for (let r = 0; r < k; r++) {
                rotated = this.rot90(rotated, cellSize);
            }
            for (let i = 0; i < cell.length; i++) {
                cell[i] += rotated[i];
            }
        }
        
        // Normalize
        for (let i = 0; i < cell.length; i++) {
            cell[i] /= 4;
        }
        
        return this.tile(cell, cellSize, cellSize, size);
    },
    
    /**
     * Simetrizar para grupos hexagonales usando promediado de rotaciones
     */
    symmetrizeHexagonal(base, size, order, withReflection = false) {
        const cx = size / 2;
        const cy = size / 2;
        const angle = 2 * Math.PI / order;
        
        const result = new Float32Array(size * size);
        
        // Promediar rotaciones
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                let sum = 0;
                const dx = x - cx;
                const dy = y - cy;
                
                for (let k = 0; k < order; k++) {
                    const theta = k * angle;
                    const cos_t = Math.cos(theta);
                    const sin_t = Math.sin(theta);
                    
                    const rx = cos_t * dx - sin_t * dy + cx;
                    const ry = sin_t * dx + cos_t * dy + cy;
                    
                    const ix = Math.min(Math.max(Math.floor(rx), 0), size - 1);
                    const iy = Math.min(Math.max(Math.floor(ry), 0), size - 1);
                    
                    sum += base[iy * size + ix];
                }
                
                result[y * size + x] = sum / order;
            }
        }
        
        if (withReflection) {
            // Añadir reflexión vertical
            const reflected = this.flipLR(result, size, size);
            for (let i = 0; i < result.length; i++) {
                result[i] = (result[i] + reflected[i]) / 2;
            }
        }
        
        return result;
    },
    
    /**
     * p3: Rotación 120° (C3)
     */
    generateP3(size, rng, motifSize = 64) {
        const cellSize = motifSize * 2;
        const base = this.createMotif(cellSize, rng);
        const cell = this.symmetrizeHexagonal(base, cellSize, 3, false);
        
        // Normalizar
        let max = 0;
        for (let i = 0; i < cell.length; i++) {
            if (cell[i] > max) max = cell[i];
        }
        if (max > 0) {
            for (let i = 0; i < cell.length; i++) {
                cell[i] /= max;
            }
        }
        
        return this.tile(cell, cellSize, cellSize, size);
    },
    
    /**
     * p3m1: C3 + reflexiones (D3)
     */
    generateP3M1(size, rng, motifSize = 64) {
        const cellSize = motifSize * 2;
        const base = this.createMotif(cellSize, rng);
        const cell = this.symmetrizeHexagonal(base, cellSize, 3, true);
        
        let max = 0;
        for (let i = 0; i < cell.length; i++) {
            if (cell[i] > max) max = cell[i];
        }
        if (max > 0) {
            for (let i = 0; i < cell.length; i++) {
                cell[i] /= max;
            }
        }
        
        return this.tile(cell, cellSize, cellSize, size);
    },
    
    /**
     * p31m: C3 + reflexiones entre centros (D3)
     */
    generateP31M(size, rng, motifSize = 64) {
        const cellSize = motifSize * 2;
        const base = this.createMotif(cellSize, rng);
        
        // Similar a p3m1 pero con reflexión horizontal
        const rotated = this.symmetrizeHexagonal(base, cellSize, 3, false);
        const reflected = this.flipUD(rotated, cellSize, cellSize);
        
        const cell = new Float32Array(cellSize * cellSize);
        for (let i = 0; i < cell.length; i++) {
            cell[i] = (rotated[i] + reflected[i]) / 2;
        }
        
        let max = 0;
        for (let i = 0; i < cell.length; i++) {
            if (cell[i] > max) max = cell[i];
        }
        if (max > 0) {
            for (let i = 0; i < cell.length; i++) {
                cell[i] /= max;
            }
        }
        
        return this.tile(cell, cellSize, cellSize, size);
    },
    
    /**
     * p6: Rotación 60° (C6)
     */
    generateP6(size, rng, motifSize = 64) {
        const cellSize = motifSize * 2;
        const base = this.createMotif(cellSize, rng);
        const cell = this.symmetrizeHexagonal(base, cellSize, 6, false);
        
        let max = 0;
        for (let i = 0; i < cell.length; i++) {
            if (cell[i] > max) max = cell[i];
        }
        if (max > 0) {
            for (let i = 0; i < cell.length; i++) {
                cell[i] /= max;
            }
        }
        
        return this.tile(cell, cellSize, cellSize, size);
    },
    
    /**
     * p6m: C6 + reflexiones (D6)
     */
    generateP6M(size, rng, motifSize = 64) {
        const cellSize = motifSize * 2;
        const base = this.createMotif(cellSize, rng);
        const cell = this.symmetrizeHexagonal(base, cellSize, 6, true);
        
        let max = 0;
        for (let i = 0; i < cell.length; i++) {
            if (cell[i] > max) max = cell[i];
        }
        if (max > 0) {
            for (let i = 0; i < cell.length; i++) {
                cell[i] /= max;
            }
        }
        
        return this.tile(cell, cellSize, cellSize, size);
    },
    
    /**
     * Generar patrón para cualquier grupo
     */
    generateFromGenerators(groupName, size, options = {}) {
        const rng = options.rng || Math.random;
        const motifSize = options.motifSize || Math.floor(size / 4);
        
        const generators = {
            'p1': () => this.generateP1(size, rng, motifSize),
            'p2': () => this.generateP2(size, rng, motifSize),
            'pm': () => this.generatePM(size, rng, motifSize),
            'pg': () => this.generatePG(size, rng, motifSize),
            'cm': () => this.generateCM(size, rng, motifSize),
            'pmm': () => this.generatePMM(size, rng, motifSize),
            'pmg': () => this.generatePMG(size, rng, motifSize),
            'pgg': () => this.generatePGG(size, rng, motifSize),
            'cmm': () => this.generateCMM(size, rng, motifSize),
            'p4': () => this.generateP4(size, rng, motifSize),
            'p4m': () => this.generateP4M(size, rng, motifSize),
            'p4g': () => this.generateP4G(size, rng, motifSize),
            'p3': () => this.generateP3(size, rng, motifSize),
            'p3m1': () => this.generateP3M1(size, rng, motifSize),
            'p31m': () => this.generateP31M(size, rng, motifSize),
            'p6': () => this.generateP6(size, rng, motifSize),
            'p6m': () => this.generateP6M(size, rng, motifSize),
        };
        
        const generator = generators[groupName];
        if (!generator) {
            console.error(`Grupo ${groupName} no encontrado`);
            return new Float32Array(size * size);
        }
        
        return generator();
    }
};

// Exportar para uso global
if (typeof window !== 'undefined') {
    window.PatternGenerator = PatternGenerator;
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = PatternGenerator;
}
