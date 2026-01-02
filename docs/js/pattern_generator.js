/**
 * Generador de patrones de wallpaper usando generadores de grupo
 * 
 * Este módulo genera patrones aplicando sistemáticamente los elementos
 * del grupo de simetría a un motivo fundamental.
 */

const PatternGenerator = {
    
    /**
     * Generar patrón aplicando elementos del grupo a un motivo
     * @param {string} groupName - Nombre del grupo (p1, p2, pm, etc.)
     * @param {number} size - Tamaño del canvas
     * @param {Object} options - Opciones de generación
     * @returns {Float32Array} - Patrón normalizado
     */
    generateFromGenerators(groupName, size, options = {}) {
        const group = WallpaperGroups[groupName];
        if (!group) {
            console.error(`Grupo ${groupName} no encontrado`);
            return new Float32Array(size * size);
        }
        
        const rng = options.rng || Math.random;
        const cellsX = options.cellsX || 3;
        const cellsY = options.cellsY || 3;
        
        // 1. Crear motivo fundamental asimétrico
        const cellWidth = Math.floor(size / cellsX);
        const cellHeight = Math.floor(size / cellsY);
        const fundamentalDomain = this.createFundamentalDomain(
            cellWidth, cellHeight, group, rng
        );
        
        // 2. Generar elementos del grupo desde generadores
        const groupElements = this.generateGroupElements(group);
        
        // 3. Aplicar cada elemento al motivo y combinar
        const pattern = new Float32Array(size * size);
        
        // Llenar el patrón aplicando simetrías
        for (let cellY = 0; cellY < cellsY; cellY++) {
            for (let cellX = 0; cellX < cellsX; cellX++) {
                this.fillCell(
                    pattern, size,
                    cellX * cellWidth, cellY * cellHeight,
                    cellWidth, cellHeight,
                    fundamentalDomain, groupElements, group
                );
            }
        }
        
        return this.normalizePattern(pattern);
    },
    
    /**
     * Crear el dominio fundamental (motivo asimétrico base)
     */
    createFundamentalDomain(width, height, group, rng) {
        // El tamaño del dominio fundamental depende del orden del grupo puntual
        const order = group.pointGroupOrder || 1;
        
        // Dividir la celda según el orden
        let domainWidth, domainHeight;
        
        if (group.hasReflection) {
            // Con reflexiones, el dominio es más pequeño
            domainWidth = Math.floor(width / 2);
            domainHeight = Math.floor(height / 2);
        } else {
            // Solo rotaciones
            const rotOrder = group.rotationOrder || 1;
            domainWidth = Math.floor(width / rotOrder);
            domainHeight = height;
        }
        
        domainWidth = Math.max(domainWidth, 10);
        domainHeight = Math.max(domainHeight, 10);
        
        // Crear motivo asimétrico usando Gaussianas aleatorias
        const domain = new Float32Array(domainWidth * domainHeight);
        const numBlobs = 3 + Math.floor(rng() * 3);
        
        for (let i = 0; i < numBlobs; i++) {
            const cx = rng() * domainWidth;
            const cy = rng() * domainHeight;
            const sigmaX = 5 + rng() * 15;
            const sigmaY = 5 + rng() * 15;
            const amplitude = 0.3 + rng() * 0.7;
            const angle = rng() * Math.PI;
            
            for (let y = 0; y < domainHeight; y++) {
                for (let x = 0; x < domainWidth; x++) {
                    const dx = x - cx;
                    const dy = y - cy;
                    
                    // Rotated elliptical Gaussian
                    const rx = dx * Math.cos(angle) + dy * Math.sin(angle);
                    const ry = -dx * Math.sin(angle) + dy * Math.cos(angle);
                    
                    const value = amplitude * Math.exp(
                        -(rx * rx) / (2 * sigmaX * sigmaX) 
                        - (ry * ry) / (2 * sigmaY * sigmaY)
                    );
                    
                    domain[y * domainWidth + x] += value;
                }
            }
        }
        
        return {
            data: domain,
            width: domainWidth,
            height: domainHeight
        };
    },
    
    /**
     * Generar todos los elementos del grupo desde los generadores
     */
    generateGroupElements(group) {
        const elements = [];
        
        // Identidad siempre
        elements.push({ type: 'identity', matrix: [[1, 0], [0, 1]], tx: 0, ty: 0 });
        
        const rotOrder = group.rotationOrder || 1;
        const hasReflection = group.hasReflection;
        
        // Rotaciones
        for (let i = 1; i < rotOrder; i++) {
            const angle = (2 * Math.PI * i) / rotOrder;
            elements.push({
                type: 'rotation',
                angle: angle,
                matrix: [
                    [Math.cos(angle), -Math.sin(angle)],
                    [Math.sin(angle), Math.cos(angle)]
                ],
                tx: 0, ty: 0
            });
        }
        
        // Reflexiones (para grupos diédricos)
        if (hasReflection) {
            // Reflexión vertical (eje Y)
            elements.push({
                type: 'reflection',
                axis: 'vertical',
                matrix: [[-1, 0], [0, 1]],
                tx: 0, ty: 0
            });
            
            // Combinar reflexión con rotaciones
            for (let i = 1; i < rotOrder; i++) {
                const angle = (2 * Math.PI * i) / rotOrder;
                const rotMatrix = [
                    [Math.cos(angle), -Math.sin(angle)],
                    [Math.sin(angle), Math.cos(angle)]
                ];
                const refMatrix = [[-1, 0], [0, 1]];
                
                // R × σ
                const combined = this.multiplyMatrices(rotMatrix, refMatrix);
                elements.push({
                    type: 'rotation_reflection',
                    matrix: combined,
                    tx: 0, ty: 0
                });
            }
        }
        
        return elements;
    },
    
    /**
     * Multiplicar matrices 2x2
     */
    multiplyMatrices(a, b) {
        return [
            [a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]],
            [a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]]
        ];
    },
    
    /**
     * Llenar una celda aplicando todas las simetrías al dominio fundamental
     */
    fillCell(pattern, patternSize, offsetX, offsetY, cellWidth, cellHeight, domain, elements, group) {
        const cx = cellWidth / 2;
        const cy = cellHeight / 2;
        
        for (let y = 0; y < cellHeight; y++) {
            for (let x = 0; x < cellWidth; x++) {
                const patternY = offsetY + y;
                const patternX = offsetX + x;
                
                if (patternX >= patternSize || patternY >= patternSize) continue;
                
                // Encontrar qué elemento del grupo mapea este punto al dominio fundamental
                let value = 0;
                let found = false;
                
                for (const elem of elements) {
                    // Transformar coordenadas inversamente
                    const dx = x - cx;
                    const dy = y - cy;
                    
                    // Aplicar transformación inversa
                    const det = elem.matrix[0][0] * elem.matrix[1][1] - elem.matrix[0][1] * elem.matrix[1][0];
                    const invMatrix = [
                        [elem.matrix[1][1] / det, -elem.matrix[0][1] / det],
                        [-elem.matrix[1][0] / det, elem.matrix[0][0] / det]
                    ];
                    
                    const srcX = invMatrix[0][0] * dx + invMatrix[0][1] * dy + cx;
                    const srcY = invMatrix[1][0] * dx + invMatrix[1][1] * dy + cy;
                    
                    // Verificar si cae en el dominio fundamental
                    if (srcX >= 0 && srcX < domain.width && srcY >= 0 && srcY < domain.height) {
                        const ix = Math.floor(srcX);
                        const iy = Math.floor(srcY);
                        value = domain.data[iy * domain.width + ix];
                        found = true;
                        break;
                    }
                }
                
                if (!found) {
                    // Wrap around usando módulo
                    const wx = ((x % domain.width) + domain.width) % domain.width;
                    const wy = ((y % domain.height) + domain.height) % domain.height;
                    value = domain.data[Math.floor(wy) * domain.width + Math.floor(wx)];
                }
                
                pattern[patternY * patternSize + patternX] = value;
            }
        }
    },
    
    /**
     * Normalizar patrón a rango [0, 1]
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
    },
    
    /**
     * Explicar cómo se genera el patrón para un grupo dado
     */
    explainGeneration(groupName) {
        const group = WallpaperGroups[groupName];
        if (!group) return "Grupo no encontrado";
        
        const lines = [
            `=== Generación del patrón ${groupName} ===`,
            ``,
            `Generadores: ${group.generators}`,
            `Orden del grupo puntual: ${group.pointGroupOrder}`,
            ``,
            `Proceso:`,
            `1. Crear motivo asimétrico en el dominio fundamental`,
            `   (1/${group.pointGroupOrder} de la celda unitaria)`,
            ``,
            `2. Aplicar los elementos del grupo:`,
        ];
        
        // Listar elementos
        if (group.cayleyTable && group.cayleyTable.elements) {
            for (const elem of group.cayleyTable.elements) {
                lines.push(`   - ${elem}`);
            }
        }
        
        lines.push(``);
        lines.push(`3. Repetir la celda por traslaciones t₁, t₂`);
        lines.push(``);
        lines.push(`Resultado: Patrón con simetría ${groupName}`);
        
        return lines.join('\n');
    }
};

// Exportar para uso global
if (typeof window !== 'undefined') {
    window.PatternGenerator = PatternGenerator;
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = PatternGenerator;
}

