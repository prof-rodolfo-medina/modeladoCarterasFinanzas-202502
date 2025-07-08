%% Implementación Completa del Algoritmo Hull-White en MATLAB
%  Maestría en Ciencias Computacionales y Matemáticas Aplicadas - UNIR
%  Curso: Modelización y Valoración de Derivados y Carteras en Finanzas

%% Función Principal: Calibración Hull-White
function [results] = hullWhiteCalibration(prices, varargin)
% HULLWHITECALIBRATION Implementa algoritmo Hull-White para calibrar u y d
% 
% Sintaxis:
%   results = hullWhiteCalibration(prices)
%   results = hullWhiteCalibration(prices, 'Probability', p)
%   results = hullWhiteCalibration(prices, 'ExponentialWeights', true)
%   results = hullWhiteCalibration(prices, 'Lambda', 0.94)
%
% Inputs:
%   prices - vector de precios observados
%   
% Parámetros opcionales (pares nombre-valor):
%   'Probability' - probabilidad de movimiento hacia arriba (default: 0.5)
%   'ExponentialWeights' - usar pesos exponenciales (default: false)
%   'Lambda' - factor de decaimiento para pesos exp (default: 0.94)
%   'Verbose' - mostrar resultados detallados (default: true)
%
% Outputs:
%   results - estructura con resultados completos

    % Parser de argumentos
    p = inputParser;
    addRequired(p, 'prices', @(x) isnumeric(x) && length(x) >= 2);
    addParameter(p, 'Probability', 0.5, @(x) isnumeric(x) && x > 0 && x < 1);
    addParameter(p, 'ExponentialWeights', false, @islogical);
    addParameter(p, 'Lambda', 0.94, @(x) isnumeric(x) && x > 0 && x < 1);
    addParameter(p, 'Verbose', true, @islogical);
    
    parse(p, prices, varargin{:});
    
    prices = p.Results.prices(:); % Asegurar vector columna
    prob = p.Results.Probability;
    useExpWeights = p.Results.ExponentialWeights;
    lambda = p.Results.Lambda;
    verbose = p.Results.Verbose;
    
    % Validación de entrada
    if length(prices) < 2
        error('Se necesitan al menos 2 observaciones de precios');
    end
    
    % Calcular rendimientos
    returns = diff(prices) ./ prices(1:end-1);
    n_returns = length(returns);
    n_obs = length(prices);
    
    % Estadísticas descriptivas
    if useExpWeights
        % Pesos exponenciales
        weights = lambda.^((n_returns-1):-1:0)';
        weights = weights / sum(weights);
        
        mu = sum(weights .* returns);
        sigma_squared = sum(weights .* (returns - mu).^2);
        sigma = sqrt(sigma_squared);
        
        results.weights = weights;
        results.lambda = lambda;
        results.weighted_mean = mu;
        results.weighted_std = sigma;
    else
        % Estadísticas estándar
        mu = mean(returns);
        sigma = std(returns); % MATLAB usa n-1 por defecto
    end
    
    % Calibración Hull-White
    if prob == 0.5
        % Versión estándar
        u = 1 + mu + sigma;
        d = 1 + mu - sigma;
    else
        % Versión generalizada
        u_minus_d = sigma / sqrt(prob * (1 - prob));
        d = (1 + mu) - prob * u_minus_d;
        u = d + u_minus_d;
    end
    
    % Validación
    is_valid = (d > 0) && (u > d) && (u > 1);
    
    % Estadísticas adicionales
    min_return = min(returns);
    max_return = max(returns);
    skewness_val = skewness(returns);
    kurtosis_val = kurtosis(returns);
    
    % Estructura de resultados
    results.prices = prices;
    results.returns = returns;
    results.n_observations = n_obs;
    results.n_returns = n_returns;
    results.mu = mu;
    results.sigma = sigma;
    results.u = u;
    results.d = d;
    results.probability = prob;
    results.is_valid = is_valid;
    results.ratio_ud = u / d;
    results.min_return = min_return;
    results.max_return = max_return;
    results.skewness = skewness_val;
    results.kurtosis = kurtosis_val;
    results.use_exp_weights = useExpWeights;
    
    % Mostrar resultados si verbose
    if verbose
        displayResults(results);
    end
end

%% Función para mostrar resultados
function displayResults(results)
    fprintf('\n');
    fprintf('╔══════════════════════════════════════════════════════════════╗\n');
    fprintf('║                   REPORTE HULL-WHITE                         ║\n');
    fprintf('╠══════════════════════════════════════════════════════════════╣\n');
    fprintf('║ DATOS DE ENTRADA:                                            ║\n');
    fprintf('║ • Número de observaciones: %25d ║\n', results.n_observations);
    fprintf('║ • Precio inicial: %35.4f ║\n', results.prices(1));
    fprintf('║ • Precio final: %37.4f ║\n', results.prices(end));
    fprintf('║                                                              ║\n');
    fprintf('║ ESTADÍSTICAS DE RENDIMIENTOS:                                ║\n');
    fprintf('║ • Rendimiento medio (μΔt): %25.6f ║\n', results.mu);
    fprintf('║ • Volatilidad (σ√Δt): %30.6f ║\n', results.sigma);
    fprintf('║ • Rendimiento mínimo: %30.6f ║\n', results.min_return);
    fprintf('║ • Rendimiento máximo: %30.6f ║\n', results.max_return);
    fprintf('║ • Asimetría: %39.6f ║\n', results.skewness);
    fprintf('║ • Curtosis: %40.6f ║\n', results.kurtosis);
    fprintf('║                                                              ║\n');
    fprintf('║ PARÁMETROS CALIBRADOS:                                       ║\n');
    fprintf('║ • Factor de subida (u): %28.6f ║\n', results.u);
    fprintf('║ • Factor de bajada (d): %28.6f ║\n', results.d);
    fprintf('║ • Ratio u/d: %35.6f ║\n', results.ratio_ud);
    fprintf('║ • Modelo válido: %33s ║\n', string(results.is_valid));
    if results.use_exp_weights
        fprintf('║ • Pesos exponenciales (λ): %23.4f ║\n', results.lambda);
    end
    fprintf('╚══════════════════════════════════════════════════════════════╝\n');
    fprintf('\n');
end

%% Función para construcción de árbol binomial
function tree = buildBinomialTree(S0, u, d, nPeriods)
% BUILDBINOMIALTREE Construye árbol binomial de precios
%
% Inputs:
%   S0 - precio inicial
%   u - factor de subida
%   d - factor de bajada  
%   nPeriods - número de períodos
%
% Output:
%   tree - cell array con precios en cada nodo

    tree = cell(nPeriods + 1, 1);
    
    for i = 1:(nPeriods + 1)
        tree{i} = zeros(i, 1);
        for j = 1:i
            nUp = (i-1) - (j-1);
            nDown = j-1;
            tree{i}(j) = S0 * (u^nUp) * (d^nDown);
        end
    end
end

%% Función para valoración de opciones europeas
function optionResults = priceEuropeanOption(S0, K, T, r, u, d, optionType)
% PRICEEUROPEANOPTION Valoración de opciones europeas con método binomial
%
% Inputs:
%   S0 - precio inicial del subyacente
%   K - precio de ejercicio
%   T - número de períodos hasta vencimiento
%   r - tasa libre de riesgo por período
%   u - factor de subida
%   d - factor de bajada
%   optionType - 'call' o 'put'
%
% Output:
%   optionResults - estructura con resultados de valoración

    % Validar inputs
    if ~ismember(lower(optionType), {'call', 'put'})
        error('optionType debe ser ''call'' o ''put''');
    end
    
    % Probabilidad risk-neutral
    q_star = (exp(r) - d) / (u - d);
    
    if q_star <= 0 || q_star >= 1
        error('Probabilidad risk-neutral inválida: %.4f', q_star);
    end
    
    % Inicializar vectores de resultados
    final_prices = zeros(T+1, 1);
    payoffs = zeros(T+1, 1);
    contributions = zeros(T+1, 1);
    
    option_value = 0;
    
    % Calcular para cada nodo final
    for j = 0:T
        % Precio final
        ST = S0 * (u^(T-j)) * (d^j);
        final_prices(j+1) = ST;
        
        % Payoff según tipo de opción
        if strcmpi(optionType, 'call')
            payoff = max(ST - K, 0);
        else % put
            payoff = max(K - ST, 0);
        end
        payoffs(j+1) = payoff;
        
        % Contribución al precio (usando distribución binomial)
        binom_prob = nchoosek(T, T-j) * (q_star^(T-j)) * ((1-q_star)^j);
        contribution = binom_prob * payoff;
        contributions(j+1) = contribution;
        
        option_value = option_value + contribution;
    end
    
    % Actualizar a valor presente
    option_value = option_value * exp(-r * T);
    
    % Estructura de resultados
    optionResults.option_value = option_value;
    optionResults.risk_neutral_prob = q_star;
    optionResults.final_prices = final_prices;
    optionResults.payoffs = payoffs;
    optionResults.contributions = contributions;
    optionResults.S0 = S0;
    optionResults.K = K;
    optionResults.T = T;
    optionResults.r = r;
    optionResults.option_type = optionType;
end

%% Función para análisis Long Straddle
function straddleResults = analyzeLongStraddle(S0, K, T, r, u, d, varargin)
% ANALYZELONGSTRADDLE Análisis completo de estrategia Long Straddle
%
% Inputs:
%   S0, K, T, r, u, d - parámetros estándar
%   
% Parámetros opcionales:
%   'MarketCall' - precio de call en mercado
%   'MarketPut' - precio de put en mercado
%
% Output:
%   straddleResults - estructura con análisis completo

    % Parser de argumentos
    p = inputParser;
    addRequired(p, 'S0', @isnumeric);
    addRequired(p, 'K', @isnumeric);
    addRequired(p, 'T', @isnumeric);
    addRequired(p, 'r', @isnumeric);
    addRequired(p, 'u', @isnumeric);
    addRequired(p, 'd', @isnumeric);
    addParameter(p, 'MarketCall', [], @isnumeric);
    addParameter(p, 'MarketPut', [], @isnumeric);
    
    parse(p, S0, K, T, r, u, d, varargin{:});
    
    marketCall = p.Results.MarketCall;
    marketPut = p.Results.MarketPut;
    
    % Valorar call y put
    callResults = priceEuropeanOption(S0, K, T, r, u, d, 'call');
    putResults = priceEuropeanOption(S0, K, T, r, u, d, 'put');
    
    % Costos de la estrategia
    theoreticalCost = callResults.option_value + putResults.option_value;
    
    % Puntos de equilibrio
    breakevenLower = K - theoreticalCost;
    breakevenUpper = K + theoreticalCost;
    
    % Análisis de payoff
    ST_range = linspace(0.5 * K, 1.5 * K, 100);
    straddlePayoffs = zeros(size(ST_range));
    
    for i = 1:length(ST_range)
        ST = ST_range(i);
        callPayoff = max(ST - K, 0);
        putPayoff = max(K - ST, 0);
        netPayoff = callPayoff + putPayoff - theoreticalCost;
        straddlePayoffs(i) = netPayoff;
    end
    
    % Estructura de resultados
    straddleResults.call_value = callResults.option_value;
    straddleResults.put_value = putResults.option_value;
    straddleResults.theoretical_cost = theoreticalCost;
    straddleResults.breakeven_lower = breakevenLower;
    straddleResults.breakeven_upper = breakevenUpper;
    straddleResults.max_loss = -theoreticalCost;
    straddleResults.ST_range = ST_range;
    straddleResults.straddle_payoffs = straddlePayoffs;
    straddleResults.call_results = callResults;
    straddleResults.put_results = putResults;
    
    % Comparación con mercado si disponible
    if ~isempty(marketCall) && ~isempty(marketPut)
        marketCost = marketCall + marketPut;
        straddleResults.market_cost = marketCost;
        straddleResults.market_call = marketCall;
        straddleResults.market_put = marketPut;
        straddleResults.price_difference = theoreticalCost - marketCost;
        straddleResults.call_diff = callResults.option_value - marketCall;
        straddleResults.put_diff = putResults.option_value - marketPut;
    end
end

%% Función para análisis de sensibilidad
function sensitivityResults = sensitivityAnalysis(prices, varargin)
% SENSITIVITYANALYSIS Analiza sensibilidad al tamaño de muestra
%
% Inputs:
%   prices - serie de precios completa
%   
% Parámetros opcionales:
%   'MinObs' - mínimo número de observaciones (default: 10)
%   'StepSize' - paso para incrementar observaciones (default: 5)
%
% Output:
%   sensitivityResults - tabla con resultados

    % Parser de argumentos
    p = inputParser;
    addRequired(p, 'prices', @isnumeric);
    addParameter(p, 'MinObs', 10, @isnumeric);
    addParameter(p, 'StepSize', 5, @isnumeric);
    
    parse(p, prices, varargin{:});
    
    minObs = p.Results.MinObs;
    stepSize = p.Results.StepSize;
    maxObs = length(prices);
    
    % Preasignar vectores
    nTests = floor((maxObs - minObs) / stepSize) + 1;
    results = struct();
    results.n_obs = zeros(nTests, 1);
    results.u = zeros(nTests, 1);
    results.d = zeros(nTests, 1);
    results.mu = zeros(nTests, 1);
    results.sigma = zeros(nTests, 1);
    results.ratio_ud = zeros(nTests, 1);
    results.is_valid = false(nTests, 1);
    results.skewness = zeros(nTests, 1);
    results.kurtosis = zeros(nTests, 1);
    
    idx = 1;
    for nObs = minObs:stepSize:maxObs
        % Usar últimas nObs observaciones
        subsample = prices(end-nObs+1:end);
        
        try
            hwResults = hullWhiteCalibration(subsample, 'Verbose', false);
            
            results.n_obs(idx) = nObs;
            results.u(idx) = hwResults.u;
            results.d(idx) = hwResults.d;
            results.mu(idx) = hwResults.mu;
            results.sigma(idx) = hwResults.sigma;
            results.ratio_ud(idx) = hwResults.ratio_ud;
            results.is_valid(idx) = hwResults.is_valid;
            results.skewness(idx) = hwResults.skewness;
            results.kurtosis(idx) = hwResults.kurtosis;
            
            idx = idx + 1;
        catch ME
            fprintf('Error con %d observaciones: %s\n', nObs, ME.message);
        end
    end
    
    % Recortar vectores al tamaño real
    fieldNames = fieldnames(results);
    for i = 1:length(fieldNames)
        results.(fieldNames{i}) = results.(fieldNames{i})(1:idx-1);
    end
    
    % Convertir a tabla
    sensitivityResults = struct2table(results);
end

%% Función para visualización completa
function plotHullWhiteAnalysis(hwResults, varargin)
% PLOTHULLWHITEANALYSIS Genera gráficos de análisis completo
%
% Inputs:
%   hwResults - resultados de hullWhiteCalibration
%   
% Parámetros opcionales:
%   'IncludeTree' - incluir gráfico de árbol binomial (default: true)
%   'TreePeriods' - número de períodos para árbol (default: 4)

    % Parser de argumentos
    p = inputParser;
    addRequired(p, 'hwResults', @isstruct);
    addParameter(p, 'IncludeTree', true, @islogical);
    addParameter(p, 'TreePeriods', 4, @isnumeric);
    
    parse(p, hwResults, varargin{:});
    
    includeTree = p.Results.IncludeTree;
    treePeriods = p.Results.TreePeriods;
    
    % Configurar figura
    if includeTree
        figure('Position', [100, 100, 1200, 800]);
        tiledlayout(2, 3, 'TileSpacing', 'compact');
    else
        figure('Position', [100, 100, 1000, 600]);
        tiledlayout(2, 2, 'TileSpacing', 'compact');
    end
    
    sgtitle('Análisis Hull-White - Visualización Completa', 'FontSize', 16, 'FontWeight', 'bold');
    
    % 1. Serie de precios
    nexttile;
    plot(hwResults.prices, 'b-', 'LineWidth', 2, 'Marker', 'o', 'MarkerSize', 4);
    title('Serie de Precios Históricos');
    xlabel('Período');
    ylabel('Precio');
    grid on;
    
    % 2. Serie de rendimientos
    nexttile;
    plot(hwResults.returns, 'r-', 'LineWidth', 1.5, 'Marker', 'o', 'MarkerSize', 3);
    hold on;
    yline(hwResults.mu, '--g', 'LineWidth', 2, 'DisplayName', sprintf('Media: %.4f', hwResults.mu));
    title('Rendimientos Relativos');
    xlabel('Período');
    ylabel('Rendimiento');
    legend;
    grid on;
    
    % 3. Histograma de rendimientos
    nexttile;
    histogram(hwResults.returns, 15, 'Normalization', 'pdf', 'FaceAlpha', 0.7, 'EdgeColor', 'black');
    hold on;
    
    % Ajustar distribución normal
    x = linspace(min(hwResults.returns), max(hwResults.returns), 100);
    y = normpdf(x, hwResults.mu, hwResults.sigma);
    plot(x, y, 'r-', 'LineWidth', 2, 'DisplayName', 'Normal ajustada');
    xline(hwResults.mu, '--g', 'LineWidth', 2, 'DisplayName', sprintf('Media: %.4f', hwResults.mu));
    
    title('Distribución de Rendimientos');
    xlabel('Rendimiento');
    ylabel('Densidad');
    legend;
    grid on;
    
    % 4. Q-Q plot para normalidad
    nexttile;
    qqplot(hwResults.returns);
    title('Q-Q Plot (Test de Normalidad)');
    grid on;
    
    % 5. Parámetros calibrados
    if includeTree
        nexttile;
        % Texto con parámetros
        paramText = sprintf(['Parámetros Hull-White:\n\n' ...
                           'u = %.6f\n' ...
                           'd = %.6f\n' ...
                           'μ = %.6f\n' ...
                           'σ = %.6f\n\n' ...
                           'Ratio u/d = %.4f\n' ...
                           'Válido: %s'], ...
                           hwResults.u, hwResults.d, hwResults.mu, hwResults.sigma, ...
                           hwResults.ratio_ud, string(hwResults.is_valid));
        
        text(0.1, 0.5, paramText, 'FontSize', 12, 'VerticalAlignment', 'middle', ...
             'BackgroundColor', [0.9, 0.9, 1], 'EdgeColor', 'black', 'Margin', 10);
        xlim([0, 1]);
        ylim([0, 1]);
        title('Parámetros Calibrados');
        axis off;
        
        % 6. Árbol binomial
        nexttile;
        plotBinomialTree(hwResults.prices(1), hwResults.u, hwResults.d, treePeriods);
    else
        % Si no incluye árbol, mostrar parámetros en una gráfica más grande
        nexttile([1, 2]);
        
        % Crear gráfico de barras con parámetros
        params = [hwResults.u, hwResults.d, hwResults.mu, hwResults.sigma];
        paramNames = {'u', 'd', 'μ', 'σ'};
        
        bar(params, 'FaceAlpha', 0.7);
        set(gca, 'XTickLabel', paramNames);
        title('Parámetros Calibrados Hull-White');
        ylabel('Valor');
        grid on;
        
        % Añadir valores en las barras
        for i = 1:length(params)
            text(i, params(i) + 0.01, sprintf('%.4f', params(i)), ...
                 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
        end
    end
end

%% Función para graficar árbol binomial
function plotBinomialTree(S0, u, d, periods)
% PLOTBINOMIALTREE Visualiza árbol binomial de precios

    tree = buildBinomialTree(S0, u, d, periods);
    
    hold on;
    
    % Plotear nodos
    for period = 1:(periods + 1)
        values = tree{period};
        nNodes = length(values);
        
        if nNodes == 1
            yPositions = 0;
        else
            yPositions = linspace(-period/2, period/2, nNodes);
        end
        
        scatter(repmat(period-1, nNodes, 1), yPositions, 100, 'red', 'filled', 'MarkerFaceAlpha', 0.7);
        
        % Añadir etiquetas con valores
        for j = 1:nNodes
            text(period-1 + 0.1, yPositions(j), sprintf('%.2f', values(j)), ...
                 'FontSize', 8, 'HorizontalAlignment', 'left');
        end
    end
    
    % Conectar nodos con líneas
    for period = 1:periods
        currentValues = tree{period};
        nextValues = tree{period + 1};
        
        nCurrent = length(currentValues);
        nNext = length(nextValues);
        
        if nCurrent == 1
            yPosCurrent = 0;
        else
            yPosCurrent = linspace(-((period-1)/2), (period-1)/2, nCurrent);
        end
        
        if nNext == 1
            yPosNext = 0;
        else
            yPosNext = linspace(-period/2, period/2, nNext);
        end
        
        % Conectar cada nodo actual con sus hijos
        for i = 1:nCurrent
            % Movimiento hacia arriba
            if i <= nNext - 1
                plot([period-1, period], [yPosCurrent(i), yPosNext(i)], 'b-', 'LineWidth', 1);
            end
            % Movimiento hacia abajo
            if i + 1 <= nNext
                plot([period-1, period], [yPosCurrent(i), yPosNext(i+1)], 'r-', 'LineWidth', 1);
            end
        end
    end
    
    xlabel('Período');
    ylabel('Posición');
    title(sprintf('Árbol Binomial (%d períodos)', periods));
    grid on;
    hold off;
end

%% Función para graficar payoff de Long Straddle
function plotStraddlePayoff(straddleResults)
% PLOTSTRADDLEPAYOFF Visualiza perfil de payoff del Long Straddle

    figure('Position', [100, 100, 1200, 500]);
    tiledlayout(1, 2, 'TileSpacing', 'compact');
    
    % Gráfico principal de payoff
    nexttile;
    
    plot(straddleResults.ST_range, straddleResults.straddle_payoffs, 'b-', 'LineWidth', 3);
    hold on;
    
    % Líneas de referencia
    yline(0, 'k--', 'LineWidth', 1, 'Alpha', 0.7, 'DisplayName', 'Línea de equilibrio');
    xline(straddleResults.call_results.K, ':', 'Color', [0.5, 0.5, 0.5], 'LineWidth', 1, ...
          'DisplayName', sprintf('Strike: %.0f', straddleResults.call_results.K));
    
    % Marcar puntos clave
    plot(straddleResults.breakeven_lower, 0, 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'red', ...
         'DisplayName', sprintf('BE Inferior: %.2f', straddleResults.breakeven_lower));
    plot(straddleResults.breakeven_upper, 0, 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'red', ...
         'DisplayName', sprintf('BE Superior: %.2f', straddleResults.breakeven_upper));
    plot(straddleResults.call_results.K, straddleResults.max_loss, 'rs', 'MarkerSize', 10, ...
         'MarkerFaceColor', 'red', 'DisplayName', sprintf('Pérdida máx: %.2f', straddleResults.max_loss));
    
    % Áreas de ganancia y pérdida
    payoffs = straddleResults.straddle_payoffs;
    ST_range = straddleResults.ST_range;
    
    % Zona de ganancia
    gainMask = payoffs > 0;
    if any(gainMask)
        fill([ST_range(gainMask), fliplr(ST_range(gainMask))], ...
             [payoffs(gainMask), zeros(size(payoffs(gainMask)))], ...
             'green', 'FaceAlpha', 0.3, 'EdgeColor', 'none', 'DisplayName', 'Zona ganancia');
    end
    
    % Zona de pérdida
    lossMask = payoffs < 0;
    if any(lossMask)
        fill([ST_range(lossMask), fliplr(ST_range(lossMask))], ...
             [payoffs(lossMask), zeros(size(payoffs(lossMask)))], ...
             'red', 'FaceAlpha', 0.3, 'EdgeColor', 'none', 'DisplayName', 'Zona pérdida');
    end
    
    xlabel('Precio del Subyacente al Vencimiento (ST)');
    ylabel('Ganancia/Pérdida');
    title('Perfil de Payoff - Long Straddle');
    legend('Location', 'best');
    grid on;
    
    % Segundo gráfico: comparación o componentes
    nexttile;
    
    if isfield(straddleResults, 'market_cost')
        % Comparación teórico vs mercado
        categories = {'Call', 'Put', 'Total'};
        theoretical = [straddleResults.call_value, straddleResults.put_value, straddleResults.theoretical_cost];
        market = [straddleResults.market_call, straddleResults.market_put, straddleResults.market_cost];
        
        x = 1:length(categories);
        width = 0.35;
        
        b1 = bar(x - width/2, theoretical, width, 'DisplayName', 'Hull-White', 'FaceAlpha', 0.8);
        hold on;
        b2 = bar(x + width/2, market, width, 'DisplayName', 'Mercado', 'FaceAlpha', 0.8);
        
        xlabel('Componente');
        ylabel('Precio');
        title('Comparación: Teórico vs Mercado');
        set(gca, 'XTickLabel', categories);
        legend;
        grid on;
        
        % Añadir valores en las barras
        for i = 1:length(theoretical)
            text(i - width/2, theoretical(i) + 0.1, sprintf('%.3f', theoretical(i)), ...
                 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
            text(i + width/2, market(i) + 0.1, sprintf('%.3f', market(i)), ...
                 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
        end
    else
        % Componentes individuales
        components = {'Call', 'Put', 'Costo Total'};
        values = [straddleResults.call_value, straddleResults.put_value, -straddleResults.theoretical_cost];
        colors = [0, 0.4470, 0.7410; 0.8500, 0.3250, 0.0980; 0.9290, 0.6940, 0.1250];
        
        b = bar(values, 'FaceAlpha', 0.7);
        b.FaceColor = 'flat';
        b.CData = colors;
        
        set(gca, 'XTickLabel', components);
        ylabel('Valor');
        title('Componentes del Long Straddle');
        grid on;
        
        % Añadir valores en las barras
        for i = 1:length(values)
            if values(i) >= 0
                text(i, values(i) + 0.1, sprintf('%.3f', values(i)), ...
                     'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
            else
                text(i, values(i) - 0.3, sprintf('%.3f', values(i)), ...
                     'HorizontalAlignment', 'center', 'VerticalAlignment', 'top');
            end
        end
    end
end

%% Función para graficar análisis de sensibilidad
function plotSensitivityAnalysis(sensitivityTable)
% PLOTSENSITIVITYANALYSIS Visualiza análisis de sensibilidad

    figure('Position', [100, 100, 1200, 800]);
    tiledlayout(2, 3, 'TileSpacing', 'compact');
    
    sgtitle('Análisis de Sensibilidad - Algoritmo Hull-White', 'FontSize', 16);
    
    % Factor u
    nexttile;
    plot(sensitivityTable.n_obs, sensitivityTable.u, 'b-o', 'LineWidth', 2, 'MarkerSize', 4);
    title('Factor u vs Tamaño de Muestra');
    xlabel('Número de Observaciones');
    ylabel('Factor u');
    grid on;
    
    % Factor d
    nexttile;
    plot(sensitivityTable.n_obs, sensitivityTable.d, 'r-o', 'LineWidth', 2, 'MarkerSize', 4);
    title('Factor d vs Tamaño de Muestra');
    xlabel('Número de Observaciones');
    ylabel('Factor d');
    grid on;
    
    % Volatilidad
    nexttile;
    plot(sensitivityTable.n_obs, sensitivityTable.sigma, 'g-o', 'LineWidth', 2, 'MarkerSize', 4);
    title('Volatilidad vs Tamaño de Muestra');
    xlabel('Número de Observaciones');
    ylabel('Volatilidad (σ)');
    grid on;
    
    % Ratio u/d
    nexttile;
    plot(sensitivityTable.n_obs, sensitivityTable.ratio_ud, 'm-o', 'LineWidth', 2, 'MarkerSize', 4);
    title('Ratio u/d vs Tamaño de Muestra');
    xlabel('Número de Observaciones');
    ylabel('u/d');
    grid on;
    
    % Rendimiento medio
    nexttile;
    plot(sensitivityTable.n_obs, sensitivityTable.mu, 'c-o', 'LineWidth', 2, 'MarkerSize', 4);
    title('Rendimiento Medio vs Tamaño de Muestra');
    xlabel('Número de Observaciones');
    ylabel('μ');
    grid on;
    
    % Validez del modelo
    nexttile;
    validCounts = groupsummary(sensitivityTable, 'n_obs', 'sum', 'is_valid');
    bar(validCounts.n_obs, validCounts.sum_is_valid, 'FaceAlpha', 0.7, 'FaceColor', [0.4660, 0.6740, 0.1880]);
    title('Modelos Válidos por Tamaño de Muestra');
    xlabel('Número de Observaciones');
    ylabel('Modelos Válidos');
    grid on;
end

%% Función para generar datos sintéticos
function prices = generateSyntheticData(varargin)
% GENERATESYNTHETICDATA Genera datos sintéticos para pruebas
%
% Parámetros opcionales:
%   'S0' - precio inicial (default: 100)
%   'NPeriods' - número de períodos (default: 100)
%   'Mu' - deriva del proceso (default: 0.001)
%   'Sigma' - volatilidad del proceso (default: 0.02)
%   'Seed' - semilla para reproducibilidad (default: 42)

    % Parser de argumentos
    p = inputParser;
    addParameter(p, 'S0', 100, @isnumeric);
    addParameter(p, 'NPeriods', 100, @isnumeric);
    addParameter(p, 'Mu', 0.001, @isnumeric);
    addParameter(p, 'Sigma', 0.02, @isnumeric);
    addParameter(p, 'Seed', 42, @isnumeric);
    
    parse(p, varargin{:});
    
    S0 = p.Results.S0;
    nPeriods = p.Results.NPeriods;
    mu = p.Results.Mu;
    sigma = p.Results.Sigma;
    seed = p.Results.Seed;
    
    % Establecer semilla
    rng(seed);
    
    % Generar rendimientos
    returns = normrnd(mu, sigma, nPeriods, 1);
    
    % Generar precios
    prices = zeros(nPeriods + 1, 1);
    prices(1) = S0;
    
    for i = 2:(nPeriods + 1)
        prices(i) = prices(i-1) * (1 + returns(i-1));
    end
end

%% Script principal de ejemplo
function mainExample()
% MAINEXAMPLE Ejemplo principal de uso completo

    clc;
    fprintf('================================================================================\n');
    fprintf('EJEMPLO COMPLETO: ALGORITMO HULL-WHITE\n');
    fprintf('================================================================================\n');
    
    % 1. Generar datos sintéticos
    fprintf('\n1. Generando datos sintéticos...\n');
    prices = generateSyntheticData('S0', 100, 'NPeriods', 50, 'Mu', 0.002, 'Sigma', 0.025);
    
    % 2. Calibración Hull-White
    fprintf('\n2. Calibración Hull-White...\n');
    hwResults = hullWhiteCalibration(prices);
    
    % 3. Análisis gráfico
    fprintf('\n3. Generando análisis gráfico...\n');
    plotHullWhiteAnalysis(hwResults);
    
    % 4. Valoración de opciones
    fprintf('\n4. Valoración de opciones...\n');
    
    % Parámetros de la opción
    S0 = 100;
    K = 100;
    T = 3;
    r = 0.05;
    
    % Análisis Long Straddle
    straddleResults = analyzeLongStraddle(S0, K, T, r, hwResults.u, hwResults.d, ...
                                         'MarketCall', 5.0, 'MarketPut', 4.0);
    
    fprintf('\nComparación con ejercicio original:\n');
    fprintf('Parámetros calibrados: u=%.6f, d=%.6f\n', hwResultsEx.u, hwResultsEx.d);
    fprintf('Call Hull-White: %.4f vs Original: 5.00\n', straddleEx.call_value);
    fprintf('Put Hull-White: %.4f vs Original: 4.00\n', straddleEx.put_value);
    
    plotHullWhiteAnalysis(hwResultsEx);
    plotStraddlePayoff(straddleEx);
end

%% Función auxiliar para mostrar árbol de manera textual
function displayTreeText(tree)
% DISPLAYTREETEXT Muestra árbol binomial en formato texto

    nPeriods = length(tree) - 1;
    fprintf('\nÁrbol Binomial de Precios:\n');
    fprintf('==========================\n');
    
    for i = 1:(nPeriods + 1)
        fprintf('Período %d: ', i-1);
        fprintf('%.4f ', tree{i});
        fprintf('\n');
    end
end

%% Función para validar parámetros Hull-White
function [isValid, warnings] = validateHullWhiteParams(u, d, verbose)
% VALIDATEHULLWHITEPARAMS Valida parámetros calibrados
%
% Inputs:
%   u, d - parámetros a validar
%   verbose - mostrar advertencias (default: true)
%
% Outputs:
%   isValid - booleano indicando si son válidos
%   warnings - cell array con advertencias

    if nargin < 3
        verbose = true;
    end
    
    warnings = {};
    isValid = true;
    
    % Verificar d > 0
    if d <= 0
        isValid = false;
        warnings{end+1} = sprintf('Factor d = %.6f <= 0. Modelo inválido.', d);
    end
    
    % Verificar u > d
    if u <= d
        isValid = false;
        warnings{end+1} = sprintf('Factor u = %.6f <= d = %.6f. Modelo inválido.', u, d);
    end
    
    % Verificar u > 1 (recomendado para crecimiento)
    if u <= 1
        warnings{end+1} = sprintf('Factor u = %.6f <= 1. Posible decrecimiento esperado.', u);
    end
    
    % Verificar d < 1 (recomendado para volatilidad)
    if d >= 1
        warnings{end+1} = sprintf('Factor d = %.6f >= 1. Comportamiento atípico.', d);
    end
    
    % Verificar ratio u/d razonable
    ratio = u / d;
    if ratio > 2
        warnings{end+1} = sprintf('Ratio u/d = %.4f muy alto. Alta volatilidad implícita.', ratio);
    elseif ratio < 1.1
        warnings{end+1} = sprintf('Ratio u/d = %.4f muy bajo. Baja volatilidad implícita.', ratio);
    end
    
    % Mostrar advertencias si verbose
    if verbose && ~isempty(warnings)
        fprintf('\nAdvertencias de validación:\n');
        for i = 1:length(warnings)
            fprintf('• %s\n', warnings{i});
        end
    end
end

%% Función para comparar diferentes métodos de calibración
function comparisonResults = compareCalibrationMethods(prices)
% COMPARECALIBRATIONMETHODS Compara diferentes enfoques de calibración
%
% Input:
%   prices - serie de precios
%
% Output:
%   comparisonResults - estructura con comparación

    % Método estándar
    hwStandard = hullWhiteCalibration(prices, 'Verbose', false);
    
    % Método con pesos exponenciales
    hwExponential = hullWhiteCalibration(prices, 'ExponentialWeights', true, 'Lambda', 0.94, 'Verbose', false);
    
    % Método con diferente probabilidad
    hwProb06 = hullWhiteCalibration(prices, 'Probability', 0.6, 'Verbose', false);
    hwProb04 = hullWhiteCalibration(prices, 'Probability', 0.4, 'Verbose', false);
    
    % Crear tabla de comparación
    methods = {'Estándar (p=0.5)', 'Pesos Exp (λ=0.94)', 'p=0.6', 'p=0.4'};
    u_values = [hwStandard.u, hwExponential.u, hwProb06.u, hwProb04.u];
    d_values = [hwStandard.d, hwExponential.d, hwProb06.d, hwProb04.d];
    mu_values = [hwStandard.mu, hwExponential.weighted_mean, hwProb06.mu, hwProb04.mu];
    sigma_values = [hwStandard.sigma, hwExponential.weighted_std, hwProb06.sigma, hwProb04.sigma];
    ratios = u_values ./ d_values;
    validity = [hwStandard.is_valid, hwExponential.is_valid, hwProb06.is_valid, hwProb04.is_valid];
    
    comparisonTable = table(methods', u_values', d_values', mu_values', sigma_values', ratios', validity', ...
                           'VariableNames', {'Método', 'u', 'd', 'μ', 'σ', 'u_d_ratio', 'Válido'});
    
    comparisonResults.table = comparisonTable;
    comparisonResults.standard = hwStandard;
    comparisonResults.exponential = hwExponential;
    comparisonResults.prob06 = hwProb06;
    comparisonResults.prob04 = hwProb04;
    
    % Mostrar tabla
    fprintf('\nComparación de Métodos de Calibración:\n');
    fprintf('======================================\n');
    disp(comparisonTable);
end

%% Función para análisis de convergencia
function convergenceResults = analyzeConvergence(prices, maxPeriods)
% ANALYZECONVERGENCE Analiza convergencia de parámetros con más datos
%
% Inputs:
%   prices - serie de precios completa
%   maxPeriods - máximo número de períodos para análisis (default: length(prices))

    if nargin < 2
        maxPeriods = length(prices);
    end
    
    if maxPeriods > length(prices)
        maxPeriods = length(prices);
    end
    
    % Análisis incremental
    minObs = 10;
    step = 1;
    
    nTests = maxPeriods - minObs + 1;
    results = struct();
    results.n_obs = zeros(nTests, 1);
    results.u = zeros(nTests, 1);
    results.d = zeros(nTests, 1);
    results.sigma = zeros(nTests, 1);
    results.ratio_ud = zeros(nTests, 1);
    
    idx = 1;
    for nObs = minObs:maxPeriods
        subsample = prices(1:nObs);
        
        try
            hwResults = hullWhiteCalibration(subsample, 'Verbose', false);
            
            results.n_obs(idx) = nObs;
            results.u(idx) = hwResults.u;
            results.d(idx) = hwResults.d;
            results.sigma(idx) = hwResults.sigma;
            results.ratio_ud(idx) = hwResults.ratio_ud;
            
            idx = idx + 1;
        catch
            % Skip if error
        end
    end
    
    % Recortar al tamaño real
    results.n_obs = results.n_obs(1:idx-1);
    results.u = results.u(1:idx-1);
    results.d = results.d(1:idx-1);
    results.sigma = results.sigma(1:idx-1);
    results.ratio_ud = results.ratio_ud(1:idx-1);
    
    convergenceResults = results;
    
    % Graficar convergencia
    figure('Position', [100, 100, 1200, 400]);
    tiledlayout(1, 3, 'TileSpacing', 'compact');
    
    nexttile;
    plot(results.n_obs, results.u, 'b-', 'LineWidth', 2);
    title('Convergencia de u');
    xlabel('Número de Observaciones');
    ylabel('u');
    grid on;
    
    nexttile;
    plot(results.n_obs, results.d, 'r-', 'LineWidth', 2);
    title('Convergencia de d');
    xlabel('Número de Observaciones');
    ylabel('d');
    grid on;
    
    nexttile;
    plot(results.n_obs, results.sigma, 'g-', 'LineWidth', 2);
    title('Convergencia de σ');
    xlabel('Número de Observaciones');
    ylabel('σ');
    grid on;
    
    sgtitle('Análisis de Convergencia de Parámetros Hull-White');
end

%% Función para test de backtesting
function backtestResults = backtestHullWhite(prices, trainRatio, testPeriods)
% BACKTESTHULLWHITE Realiza backtesting del modelo Hull-White
%
% Inputs:
%   prices - serie de precios completa
%   trainRatio - proporción para entrenamiento (default: 0.8)
%   testPeriods - períodos para predicción (default: 3)

    if nargin < 2
        trainRatio = 0.8;
    end
    if nargin < 3
        testPeriods = 3;
    end
    
    nTotal = length(prices);
    nTrain = floor(nTotal * trainRatio);
    
    if nTrain < 10
        error('Muy pocos datos para entrenamiento');
    end
    
    % Dividir datos
    trainPrices = prices(1:nTrain);
    testPrices = prices(nTrain+1:end);
    
    % Calibrar con datos de entrenamiento
    hwResults = hullWhiteCalibration(trainPrices, 'Verbose', false);
    
    % Generar predicciones
    S0_test = trainPrices(end);
    u = hwResults.u;
    d = hwResults.d;
    
    % Simular posibles trayectorias
    nSimulations = 1000;
    predictedPaths = zeros(nSimulations, testPeriods);
    
    for sim = 1:nSimulations
        currentPrice = S0_test;
        for period = 1:testPeriods
            if rand < 0.5
                currentPrice = currentPrice * u;
            else
                currentPrice = currentPrice * d;
            end
            predictedPaths(sim, period) = currentPrice;
        end
    end
    
    % Estadísticas de predicción
    predictedMean = mean(predictedPaths, 1);
    predictedStd = std(predictedPaths, 0, 1);
    predictedCI_lower = prctile(predictedPaths, 5, 1);
    predictedCI_upper = prctile(predictedPaths, 95, 1);
    
    % Comparar con precios reales
    actualPrices = testPrices(1:min(testPeriods, length(testPrices)));
    
    % Métricas de error
    if length(actualPrices) >= testPeriods
        mse = mean((predictedMean(1:length(actualPrices)) - actualPrices').^2);
        mae = mean(abs(predictedMean(1:length(actualPrices)) - actualPrices'));
        mape = mean(abs((predictedMean(1:length(actualPrices)) - actualPrices') ./ actualPrices') * 100);
    else
        mse = NaN;
        mae = NaN;
        mape = NaN;
    end
    
    % Resultados
    backtestResults.train_size = nTrain;
    backtestResults.test_size = length(testPrices);
    backtestResults.hw_params = hwResults;
    backtestResults.predicted_mean = predictedMean;
    backtestResults.predicted_std = predictedStd;
    backtestResults.predicted_ci_lower = predictedCI_lower;
    backtestResults.predicted_ci_upper = predictedCI_upper;
    backtestResults.actual_prices = actualPrices;
    backtestResults.mse = mse;
    backtestResults.mae = mae;
    backtestResults.mape = mape;
    
    % Visualización
    figure('Position', [100, 100, 1000, 600]);
    
    % Serie completa
    subplot(2, 1, 1);
    plot(1:nTrain, trainPrices, 'b-', 'LineWidth', 2, 'DisplayName', 'Entrenamiento');
    hold on;
    if ~isempty(testPrices)
        plot(nTrain+1:nTrain+length(testPrices), testPrices, 'r-', 'LineWidth', 2, 'DisplayName', 'Test Real');
    end
    xline(nTrain, 'k--', 'LineWidth', 1, 'DisplayName', 'División Train/Test');
    title('Serie de Precios: Entrenamiento vs Test');
    xlabel('Período');
    ylabel('Precio');
    legend;
    grid on;
    
    % Predicciones vs real
    subplot(2, 1, 2);
    periods = 1:testPeriods;
    
    % Banda de confianza
    fill([periods, fliplr(periods)], [predictedCI_lower, fliplr(predictedCI_upper)], ...
         'blue', 'FaceAlpha', 0.3, 'EdgeColor', 'none', 'DisplayName', 'IC 90%');
    hold on;
    
    % Predicción media
    plot(periods, predictedMean, 'b-', 'LineWidth', 2, 'DisplayName', 'Predicción Media');
    
    % Precios reales
    if ~isempty(actualPrices)
        plot(1:length(actualPrices), actualPrices, 'ro-', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Precios Reales');
    end
    
    title('Predicciones Hull-White vs Precios Reales');
    xlabel('Período de Predicción');
    ylabel('Precio');
    legend;
    grid on;
    
    % Mostrar métricas
    if ~isnan(mse)
        fprintf('\nMétricas de Backtesting:\n');
        fprintf('MSE: %.6f\n', mse);
        fprintf('MAE: %.6f\n', mae);
        fprintf('MAPE: %.2f%%\n', mape);
    end
end

%% Función principal para ejecutar todos los ejemplos
function runAllExamples()
% RUNALLEXAMPLES Ejecuta todos los ejemplos disponibles

    fprintf('================================================================================\n');
    fprintf('SUITE COMPLETA DE EJEMPLOS - ALGORITMO HULL-WHITE\n');
    fprintf('================================================================================\n');
    
    % Generar datos para ejemplos
    prices = generateSyntheticData('NPeriods', 60, 'Sigma', 0.03);
    
    fprintf('\n[1/6] Calibración básica...\n');
    hwResults = hullWhiteCalibration(prices);
    
    fprintf('\n[2/6] Análisis gráfico completo...\n');
    plotHullWhiteAnalysis(hwResults);
    
    fprintf('\n[3/6] Comparación de métodos...\n');
    comparisonResults = compareCalibrationMethods(prices);
    
    fprintf('\n[4/6] Análisis de sensibilidad...\n');
    sensitivityTable = sensitivityAnalysis(prices, 'MinObs', 15, 'StepSize', 5);
    plotSensitivityAnalysis(sensitivityTable);
    
    fprintf('\n[5/6] Análisis de convergencia...\n');
    convergenceResults = analyzeConvergence(prices, 50);
    
    fprintf('\n[6/6] Backtesting...\n');
    backtestResults = backtestHullWhite(prices, 0.75, 5);
    
    fprintf('\n================================================================================\n');
    fprintf('SUITE COMPLETA EJECUTADA EXITOSAMENTE\n');
    fprintf('================================================================================\n');
end

%% Ejecutar ejemplo principal al cargar el archivo
if ~isdeployed && ismember('-run', {''}) % Solo ejecutar si se llama directamente
    mainExample();
end
    
    fprintf('\nResultados Long Straddle:\n');
    fprintf('Call teórica: %.4f vs Mercado: %.4f\n', straddleResults.call_value, straddleResults.market_call);
    fprintf('Put teórica: %.4f vs Mercado: %.4f\n', straddleResults.put_value, straddleResults.market_put);
    fprintf('Costo total teórico: %.4f\n', straddleResults.theoretical_cost);
    fprintf('Costo total mercado: %.4f\n', straddleResults.market_cost);
    fprintf('Diferencia: %.4f\n', straddleResults.price_difference);
    fprintf('Breakeven inferior: %.2f\n', straddleResults.breakeven_lower);
    fprintf('Breakeven superior: %.2f\n', straddleResults.breakeven_upper);
    
    % Gráfico del straddle
    plotStraddlePayoff(straddleResults);
    
    % 5. Análisis de sensibilidad
    fprintf('\n5. Análisis de sensibilidad...\n');
    sensitivityTable = sensitivityAnalysis(prices, 'MinObs', 15, 'StepSize', 3);
    plotSensitivityAnalysis(sensitivityTable);
    
    fprintf('\nResumen de sensibilidad:\n');
    disp(summary(sensitivityTable));
    
    fprintf('\n================================================================================\n');
    fprintf('ANÁLISIS COMPLETADO\n');
    fprintf('================================================================================\n');
    
    % Ejemplo adicional con datos del ejercicio original
    fprintf('\n================================================================================\n');
    fprintf('EJEMPLO CON DATOS DEL EJERCICIO ORIGINAL\n');
    fprintf('================================================================================\n');
    
    % Datos de ejemplo del documento
    exercisePrices = [100, 102, 98, 105, 103, 99, 107, 104, 101, 108, 106, 103];
    
    hwResultsEx = hullWhiteCalibration(exercisePrices);
    
    % Comparar con parámetros dados en el ejercicio
    straddleEx = analyzeLongStraddle(100, 100, 3, 0.05, hwResultsEx.u, hwResultsEx.d, ...
                                    