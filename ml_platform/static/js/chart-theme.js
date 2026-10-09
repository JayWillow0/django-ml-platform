/* =========================================================================
   chart-theme.js — MLPlatform ECharts 主题覆盖层
   -------------------------------------------------------------------------
   后端 ml_utils.py 返回完整 ECharts option（前端直接 setOption）。
   本模块在前端 setOption 之前调用 applyChartTheme(option)，注入统一色板、
   字体、坐标轴、tooltip 样式 —— 让全站图表视觉与 UI 一致，且无需改动后端。

   用法：
       chart.setOption(applyChartTheme(resp.echarts));

   合并规则：默认主题字段在前，后端 option 已有的字段覆盖之。
   图表颜色由本模块统一收口，避免模板和后端散落硬编码。
   ========================================================================= */
(function () {
    var BRAND_SERIES_PALETTE = ['#1A8BFB', '#36BC8A', '#C3C8D0', '#E1E4E7', '#D8DCE0'];
    var SINGLE_BAR_COLOR = '#141818';
    var OVERFLOW_BAR_COLOR = '#E1E4E7';
    var HEATMAPS = {
        rdBu: ['#2166AC', '#67A9CF', '#F7F7F7', '#EF8A62', '#B2182B'],
        rdBuReverse: ['#B2182B', '#EF8A62', '#F7F7F7', '#67A9CF', '#2166AC'],
        brandCool: ['#1A8BFB', '#C3C8D0', '#F7F7F7', '#E1E4E7', '#141818'],
        grayScale: ['#141818', '#C3C8D0', '#F7F7F7', '#E1E4E7', '#D8DCE0']
    };
    var FONT_FAMILY = "'IBM Plex Sans','PingFang SC','Microsoft YaHei',sans-serif";
    var AXIS_LINE = '#e2e8f0';
    var AXIS_LABEL = '#64748b';
    var SPLIT_LINE = '#f1f5f9';

    // 浅合并：src 的字段覆盖 target（仅一层），不递归。
    function merge(target, src) {
        if (!src) return target;
        target = target || {};
        for (var k in src) {
            if (Object.prototype.hasOwnProperty.call(src, k)) target[k] = src[k];
        }
        return target;
    }

    function asArray(value) {
        if (!value) return [];
        return Array.isArray(value) ? value : [value];
    }

    function hasHeatmap(option) {
        var series = asArray(option.series);
        for (var i = 0; i < series.length; i++) {
            if (series[i] && series[i].type === 'heatmap') return true;
        }
        return false;
    }

    function getCategoryCount(option) {
        var axes = asArray(option.xAxis).concat(asArray(option.yAxis));
        for (var i = 0; i < axes.length; i++) {
            var axis = axes[i];
            if (axis && axis.type === 'category' && Array.isArray(axis.data)) {
                return axis.data.length;
            }
        }
        return 0;
    }

    function firstAxis(option, key) {
        var axes = asArray(option[key]);
        return axes.length ? axes[0] : {};
    }

    function getTitleText(option) {
        var titles = asArray(option.title);
        if (!titles.length) return '';
        return (titles[0] && titles[0].text ? String(titles[0].text) : '').toLowerCase();
    }

    function isHorizontalValueBar(option) {
        var xAxis = firstAxis(option, 'xAxis');
        var yAxis = firstAxis(option, 'yAxis');
        return (!xAxis.type || xAxis.type === 'value') && yAxis.type === 'category';
    }

    function isModelPerformanceChart(option, meta) {
        if (meta && meta.chartType === 'model_comparison') return true;
        var title = getTitleText(option);
        return title.indexOf('模型') >= 0 && title.indexOf('性能') >= 0
            || title.indexOf('model') >= 0 && title.indexOf('performance') >= 0;
    }

    function isRankedMetricChart(option, meta) {
        if (meta && (
            meta.chartType === 'correlation'
            || meta.chartType === 'feature_importance'
            || meta.chartType === 'model_comparison'
        )) return true;
        var title = getTitleText(option);
        return title.indexOf('correlation') >= 0
            || title.indexOf('importance') >= 0
            || title.indexOf('performance') >= 0;
    }

    function isOptimizationChart(option, meta) {
        if (meta && meta.chartType === 'optimization') return true;
        var title = getTitleText(option);
        return title.indexOf('optuna') >= 0 || title.indexOf('optimization') >= 0;
    }

    function getDataCount(series) {
        return Array.isArray(series.data) ? series.data.length : 0;
    }

    function normalizeColor(color) {
        return typeof color === 'string' ? color.toUpperCase() : '';
    }

    function isBrandColor(color) {
        color = normalizeColor(color);
        if (!color) return false;
        for (var i = 0; i < BRAND_SERIES_PALETTE.length; i++) {
            if (normalizeColor(BRAND_SERIES_PALETTE[i]) === color) return true;
        }
        return color === normalizeColor(SINGLE_BAR_COLOR) || color === normalizeColor(OVERFLOW_BAR_COLOR);
    }

    function getDataValue(item) {
        if (item && typeof item === 'object' && !Array.isArray(item)) return item.value;
        return item;
    }

    function paletteColor(index, overflowColor) {
        if (index < BRAND_SERIES_PALETTE.length) return BRAND_SERIES_PALETTE[index];
        return overflowColor || OVERFLOW_BAR_COLOR;
    }

    function getHeatmapPalette(name) {
        return HEATMAPS[name] || HEATMAPS.rdBu;
    }

    function colorBarData(series, colorForIndex) {
        var data = Array.isArray(series.data) ? series.data : [];
        series.data = data.map(function (item, index) {
            var color = colorForIndex(index);
            if (item && typeof item === 'object' && !Array.isArray(item)) {
                item.itemStyle = merge({ color: color }, item.itemStyle);
                item.itemStyle.color = color;
                return item;
            }
            return {
                value: item,
                itemStyle: { color: color }
            };
        });
    }

    function colorBarDataByRank(series, preferHigherIndexOnTie, rankBy) {
        var data = Array.isArray(series.data) ? series.data : [];
        var ranked = data.map(function (item, index) {
            var value = parseFloat(getDataValue(item));
            if (!isNaN(value) && rankBy === 'abs') value = Math.abs(value);
            return { index: index, value: isNaN(value) ? Number.NEGATIVE_INFINITY : value };
        }).sort(function (a, b) {
            if (b.value === a.value) {
                return preferHigherIndexOnTie ? b.index - a.index : a.index - b.index;
            }
            return b.value - a.value;
        });
        var colorsByIndex = {};
        for (var i = 0; i < ranked.length; i++) {
            colorsByIndex[ranked[i].index] = paletteColor(i);
        }
        colorBarData(series, function (index) { return colorsByIndex[index] || OVERFLOW_BAR_COLOR; });
    }

    function colorGroupedBarsByRank(barSeries) {
        var maxDataCount = 0;
        for (var i = 0; i < barSeries.length; i++) {
            maxDataCount = Math.max(maxDataCount, getDataCount(barSeries[i].series));
        }

        var colorsBySeries = [];
        for (var j = 0; j < barSeries.length; j++) colorsBySeries[j] = {};

        for (var dataIndex = 0; dataIndex < maxDataCount; dataIndex++) {
            var ranked = [];
            for (var seriesIndex = 0; seriesIndex < barSeries.length; seriesIndex++) {
                var data = Array.isArray(barSeries[seriesIndex].series.data)
                    ? barSeries[seriesIndex].series.data
                    : [];
                var rawValue = dataIndex < data.length ? getDataValue(data[dataIndex]) : null;
                var value = parseFloat(rawValue);
                ranked.push({
                    seriesIndex: seriesIndex,
                    value: isNaN(value) ? Number.NEGATIVE_INFINITY : value
                });
            }
            ranked.sort(function (a, b) { return b.value - a.value; });
            for (var rank = 0; rank < ranked.length; rank++) {
                colorsBySeries[ranked[rank].seriesIndex][dataIndex] = paletteColor(rank);
            }
        }

        for (var k = 0; k < barSeries.length; k++) {
            colorBarData(barSeries[k].series, function (index) {
                return colorsBySeries[k][index] || OVERFLOW_BAR_COLOR;
            });
        }
    }

    function colorGroupedBarsBySeries(barSeries) {
        for (var i = 0; i < barSeries.length; i++) {
            colorBarData(barSeries[i].series, function () {
                return paletteColor(i);
            });
        }
    }

    function styleBarSeries(option, meta) {
        var series = asArray(option.series);
        var barSeries = [];
        var hasStackedBars = false;
        for (var i = 0; i < series.length; i++) {
            if (series[i] && series[i].type === 'bar') {
                barSeries.push({ series: series[i], index: i });
                if (series[i].stack) hasStackedBars = true;
            }
        }
        if (!barSeries.length) return;

        var categoryCount = getCategoryCount(option);
        var barMode = meta && meta.barMode;
        if (!barMode && barSeries.length === 1 && isHorizontalValueBar(option) && isRankedMetricChart(option, meta)) {
            barMode = 'rank';
        }
        var useCategoryColors = barMode === 'category' || (!barMode && barSeries.length === 1 && !hasStackedBars);
        var singleCategory = categoryCount === 1 || (barSeries.length === 1 && getDataCount(barSeries[0].series) === 1);

        if (barMode === 'group-rank') {
            for (var g = 0; g < barSeries.length; g++) {
                barSeries[g].series.itemStyle = merge(
                    { borderRadius: [4, 4, 0, 0] },
                    barSeries[g].series.itemStyle
                );
            }
            colorGroupedBarsByRank(barSeries);
            return;
        }

        if (barMode === 'group-series') {
            for (var gs = 0; gs < barSeries.length; gs++) {
                barSeries[gs].series.itemStyle = merge(
                    { borderRadius: [4, 4, 0, 0] },
                    barSeries[gs].series.itemStyle
                );
            }
            colorGroupedBarsBySeries(barSeries);
            return;
        }

        for (var j = 0; j < barSeries.length; j++) {
            var s = barSeries[j].series;
            s.itemStyle = merge({ borderRadius: [4, 4, 0, 0] }, s.itemStyle);

            if (barMode === 'rank') {
                colorBarDataByRank(s, isHorizontalValueBar(option), meta && meta.rankBy);
            } else if (singleCategory) {
                s.itemStyle.color = SINGLE_BAR_COLOR;
                if (Array.isArray(s.data)) {
                    colorBarData(s, function () { return SINGLE_BAR_COLOR; });
                }
            } else if (useCategoryColors) {
                colorBarData(s, function (index) { return paletteColor(index); });
            } else {
                if (!s.itemStyle.color) {
                    s.itemStyle.color = paletteColor(j);
                }
            }
        }
    }

    function semanticLineColor(series, index) {
        var name = (series.name || '').toLowerCase();
        if (name.indexOf('best') >= 0 || name.indexOf('最佳') >= 0) return BRAND_SERIES_PALETTE[0];
        if (name.indexOf('trial') >= 0 || name.indexOf('score') >= 0 || name.indexOf('试验') >= 0) {
            return BRAND_SERIES_PALETTE[2];
        }
        return paletteColor(index);
    }

    function semanticScatterColor(series, index) {
        var name = (series.name || '').toLowerCase();
        if (name.indexOf('normal') >= 0 || name.indexOf('正常') >= 0) return BRAND_SERIES_PALETTE[1];
        if (name.indexOf('anomaly') >= 0 || name.indexOf('异常') >= 0) return BRAND_SERIES_PALETTE[0];
        if (isBrandColor(series.itemStyle && series.itemStyle.color)) return series.itemStyle.color;
        return paletteColor(index);
    }

    function styleLineAndScatter(option) {
        var series = asArray(option.series);
        var lineIndex = 0;
        var scatterIndex = 0;
        for (var i = 0; i < series.length; i++) {
            var s = series[i];
            if (!s || (s.type !== 'line' && s.type !== 'scatter')) continue;
            var color = s.type === 'line'
                ? semanticLineColor(s, lineIndex)
                : semanticScatterColor(s, scatterIndex);
            if (s.type === 'line') lineIndex++;
            else scatterIndex++;
            s.itemStyle = merge({ color: color }, s.itemStyle);
            s.itemStyle.color = color;
            if (s.type === 'line') {
                s.lineStyle = merge({ color: color, width: i === 0 ? 1.8 : 2.2 }, s.lineStyle);
                s.lineStyle.color = color;
                if (s.areaStyle && !s.areaStyle.color) {
                    s.areaStyle.color = 'rgba(26,139,251,0.14)';
                }
            }
        }
    }

    function collectSeriesNumbers(option, seriesType) {
        var values = [];
        var series = asArray(option.series);
        for (var i = 0; i < series.length; i++) {
            var s = series[i];
            if (!s || s.type !== seriesType || !Array.isArray(s.data)) continue;
            for (var j = 0; j < s.data.length; j++) {
                var value = parseFloat(getDataValue(s.data[j]));
                if (!isNaN(value) && isFinite(value)) values.push(value);
            }
        }
        return values;
    }

    function styleOptimizationAxis(option, meta) {
        if (!isOptimizationChart(option, meta)) return;
        var yAxis = firstAxis(option, 'yAxis');
        if (!yAxis || typeof yAxis !== 'object') return;

        var values = collectSeriesNumbers(option, 'line');
        if (!values.length) return;

        var minValue = Math.min.apply(null, values);
        var maxValue = Math.max.apply(null, values);
        var span = maxValue - minValue;
        var pad = span > 0 ? Math.max(span * 0.18, 0.01) : Math.max(Math.abs(maxValue) * 0.02, 0.01);
        var axisMin = minValue - pad;
        var axisMax = maxValue + pad;
        var boundedScore = minValue >= 0 && maxValue <= 1;
        if (boundedScore) {
            axisMin = Math.max(0, axisMin);
            axisMax = Math.min(1, axisMax);
        }
        if (axisMin === axisMax) {
            axisMin -= 0.01;
            axisMax += 0.01;
        }
        yAxis.min = Number(axisMin.toFixed(4));
        yAxis.max = Number(axisMax.toFixed(4));
        yAxis.scale = true;
    }

    function styleVisualMap(option, meta) {
        if (!option.visualMap) return;
        var visualMaps = asArray(option.visualMap);
        var paletteName = meta && meta.heatmapPalette;
        for (var i = 0; i < visualMaps.length; i++) {
            var vm = visualMaps[i];
            if (!vm || typeof vm !== 'object') continue;
            vm.textStyle = merge(
                { fontFamily: FONT_FAMILY, color: AXIS_LABEL, fontSize: 11 },
                vm.textStyle
            );
            if (hasHeatmap(option)) {
                vm.inRange = merge({ color: getHeatmapPalette(paletteName) }, vm.inRange);
                vm.inRange.color = getHeatmapPalette(paletteName);
            }
        }
    }

    function styleAxis(option, key) {
        var axis = option[key];
        if (!axis) return;
        var list = Array.isArray(axis) ? axis : [axis];
        for (var i = 0; i < list.length; i++) {
            var a = list[i];
            if (!a || typeof a !== 'object') continue;
            a.axisLine = merge({ lineStyle: { color: AXIS_LINE } }, a.axisLine);
            a.axisTick = merge({ show: false }, a.axisTick);
            a.axisLabel = merge({ color: AXIS_LABEL, fontFamily: FONT_FAMILY, fontSize: 11 }, a.axisLabel);
            a.splitLine = merge({ lineStyle: { color: SPLIT_LINE } }, a.splitLine);
        }
    }

    function applyPalette(option, meta) {
        if (!option || typeof option !== 'object') return option;
        meta = merge(merge({}, option.mlTheme), meta || {});

        option.color = BRAND_SERIES_PALETTE.slice();
        styleBarSeries(option, meta || {});
        styleLineAndScatter(option);
        styleOptimizationAxis(option, meta || {});
        return option;
    }

    window.MLChartTheme = {
        palette: BRAND_SERIES_PALETTE.slice(),
        singleBarColor: SINGLE_BAR_COLOR,
        overflowBarColor: OVERFLOW_BAR_COLOR,
        heatmaps: HEATMAPS,
        getHeatmapPalette: getHeatmapPalette,
        applyPalette: applyPalette
    };

    window.applyChartTheme = function (option, meta) {
        if (!option || typeof option !== 'object') return option;
        var themeMeta = merge(merge({}, option.mlTheme), meta || {});

        applyPalette(option, themeMeta);

        option.textStyle = merge(
            { fontFamily: FONT_FAMILY, color: '#334155', fontSize: 12 },
            option.textStyle
        );

        if (option.title) {
            option.title.textStyle = merge(
                { fontFamily: FONT_FAMILY, fontSize: 15, fontWeight: 600, color: '#0f172a' },
                option.title.textStyle
            );
            if (option.title.subtext) {
                option.title.subtextStyle = merge(
                    { fontFamily: FONT_FAMILY, color: '#64748b', fontSize: 12 },
                    option.title.subtextStyle
                );
            }
        }

        option.tooltip = merge({
            backgroundColor: '#0f172a',
            borderColor: 'transparent',
            borderWidth: 0,
            padding: [8, 12],
            textStyle: { color: '#fff', fontFamily: FONT_FAMILY, fontSize: 12 },
            axisPointer: { lineStyle: { color: '#cbd5e1' }, crossStyle: { color: '#cbd5e1' } }
        }, option.tooltip);

        option.legend = merge({
            textStyle: { fontFamily: FONT_FAMILY, color: '#475569', fontSize: 12 },
            icon: 'circle',
            itemGap: 16
        }, option.legend);

        styleAxis(option, 'xAxis');
        styleAxis(option, 'yAxis');

        option.grid = merge(
            { left: '3%', right: '4%', bottom: '3%', top: 48, containLabel: true },
            option.grid
        );

        styleVisualMap(option, themeMeta);

        return option;
    };
})();
