var isAPIEnabled = false;
if (window.FileReader && window.File) {
    isAPIEnabled = true;
} else {
    alert('обновите браузер!');
}

var array = [],
    lanes = [],
    items = [];
var timeBegin = 0,
    timeEnd = 0,
    laneLength = 0;

var margin = [40, 20, 20, 150],
    width = 1000 - margin[1] - margin[3],
    height = 600 - margin[0] - margin[2],
    timelineHeight = laneLength * 30 + 100,
    zoomedAreaHeight = height - timelineHeight - 80;

function readFile(obj) {

    var file = obj.files[0];
    var reader = new FileReader();
    reader.onload = function(e) {
        var text = e.target.result;
        var lines = text.split('\n');

        for (var i = 0; i < lines.length; i++) {
            var k = lines[i].split(/(\d*\.?\d+), (\d+), (\w+), (\w+)/);
            array.
            push({
                'time': k[1],
                'id': k[2],
                'lane': k[3],
                'type': k[4]
            });
            if (lanes.indexOf(k[3]) == -1)
                lanes.push(k[3]);
            if (timeEnd < k[1])
                timeEnd = k[1];

        }
        laneLength = lanes.length;

        for (var i = 0; i < array.length; i++) {
            if (array[i]['type'] == 'begin') {
                var item = {
                    'lane': lanes.findIndex(x => x == array[i]['lane']),
                    'id': array[i]['id'],
                    'begin': array[i]['time']
                };
                items.push(findBeginEndPair(i, item));
            }
        }
    };
    reader.readAsText(file);
    setTimeout(() => {
        startRender();
        showUtilization();
    }, 10);
}

function findBeginEndPair(i, item) {
    var j = i + 1;
    while (
        (array[i]['id'] != array[j]['id']) && (array[i]['lane'] != array[j]['lane']) &&
        (j < array.length) && (array[j]['type'] != 'end')) {
        j++;
    }
    return {
        'lane': item['lane'],
        'id': item['id'],
        'begin': +item['begin'],
        'end': +array[j]['time']
    };
}

function showUtilization() {
    var text = document.getElementById('out');
    var localTimeSum = 0;
    for (let i = 0; i < items.length; i++)
        localTimeSum += items[i]['end'] - items[i]['begin'];
    text.innerText = "Utilization: " + ((localTimeSum / ((lanes.length - 1) * (timeEnd - timeBegin))) * 100).toFixed(2) + '%';
}

function startRender() {
    var div = d3.select("body").append("div")
        .attr("class", "tooltip")
        .style("opacity", 0);
    var xTimeScale = d3.scale.linear()
        .domain([timeBegin, timeEnd])
        .range([0, width]);
    var xWidthScale = d3.scale.linear()
        .range([0, width]);
    var y1 = d3.scale.linear()
        .domain([0, laneLength])
        .range([0, zoomedAreaHeight]);
    var y2 = d3.scale.linear()
        .domain([0, laneLength])
        .range([0, timelineHeight]);

    var chart = d3.select("body")
        .append("svg")
        .attr("width", width + margin[1] + margin[3])
        .attr("height", height + margin[0] + margin[2]);

    chart.append("defs").append("clipPath")
        .attr("id", "clip")
        .append("rect")
        .attr("width", width)
        .attr("height", zoomedAreaHeight);

    var zoomedArea = chart.append("g")
        .attr("transform", "translate(" + margin[3] + "," + margin[0] + ")")
        .attr("width", width)
        .attr("height", zoomedAreaHeight);

    var timeline = chart.append("g")
        .attr("transform", "translate(" + margin[3] + "," + (zoomedAreaHeight + margin[0]) + ")")
        .attr("width", width)
        .attr("height", timelineHeight)
        .attr("class", "timeline");

    zoomedArea.append("g").selectAll(".laneLines")
        .data(items)
        .enter().append("line")
        .attr("xWidthScale", margin[1])
        .attr("y1", function(d) {
            return y1(d.lane);
        })
        .attr("x2", width)
        .attr("y2", function(d) {
            return y1(d.lane);
        });

    zoomedArea.append("g").selectAll(".laneText")
        .data(lanes)
        .enter().append("text")
        .text(function(d) {
            return d;
        })
        .attr("x", -margin[1])
        .attr("y", function(d, i) {
            return y1(i + .5);
        })
        .attr("dy", ".5ex")
        .attr("text-anchor", "end")
        .attr("class", "laneText");

    timeline.append("g").selectAll(".laneLines")
        .data(items)
        .enter().append("line")
        .attr("xWidthScale", margin[1])
        .attr("y1", function(d) {
            return y2(d.lane);
        })
        .attr("x2", width)
        .attr("y2", function(d) {
            return y2(d.lane);
        })
        .attr("stroke", "lightgray");

    timeline.append("g").selectAll(".laneText")
        .data(lanes)
        .enter().append("text")
        .text(function(d) {
            return d;
        })
        .attr("x", -margin[1])
        .attr("y", function(d, i) {
            return y2(i + .5);
        })
        .attr("dy", ".5ex")
        .attr("text-anchor", "end")
        .attr("class", "laneText");

    var itemRects = zoomedArea.append("g")
        .attr("clip-path", "url(#clip)");

    timeline.append("g").selectAll("miniItems")
        .data(items)
        .enter().append("rect")
        .attr("class", function(d) {
            return "miniItem" + d.lane;
        })
        .attr("x", function(d) {
            return xTimeScale(d.begin);
        })
        .attr("y", function(d) {
            return y2(d.lane + .5) - 5;
        })
        .attr("width", function(d) {
            return xTimeScale(d.end - d.begin);
        })
        .attr("height", 20);


    var brush = d3.svg.brush()
        .x(xTimeScale)
        .on("brush", render);

    timeline.append("g")
        .attr("class", "x brush")
        .call(brush)
        .selectAll("rect")
        .attr("y", 1)
        .attr("height", timelineHeight - 1)
        .data(items);

    render();

    function render() {
        var rects, labels,
            minExtent = brush.extent()[0],
            maxExtent = brush.extent()[1],
            visItems = items.filter(function(d) {
                return d.begin < maxExtent && d.end > minExtent;
            });

        timeline.select(".brush")
            .call(brush.extent([minExtent, maxExtent]));

        xWidthScale.domain([minExtent, maxExtent]);

        rects = itemRects.selectAll("rect")
            .data(visItems, function(d) {
                return d.id;
            })
            .attr("x", function(d) {
                return xWidthScale(d.begin);
            })
            .attr("width", function(d) {
                return xWidthScale(d.end) - xWidthScale(d.begin);
            })
            .on("mouseover", function(d) {
                if (!d.id)
                    return;
                div.transition()
                    .duration(300)
                    .style("opacity", .8);
                div.html("id:" + d.id + "<br/>" + "begin: " + d.begin + "<br/>" + "end: " + d.end)
                    .style("top", (d3.event.pageY - 40) + "px").style("left", (d3.event.pageX) + "px");
            })
            .on("mouseout", function(d) {
                div.transition()
                    .duration(500)
                    .style("opacity", 0);
            });

        rects.enter().append("rect")
            .attr("class", function(d) {
                return "miniItem" + d.lane;
            })
            .attr("x", function(d) {
                return xWidthScale(d.begin);
            })
            .attr("y", function(d) {
                return y1(d.lane) + 10;
            })
            .attr("width", function(d) {
                return xWidthScale(d.end) - xWidthScale(d.begin);
            })
            .attr("height", function(d) {
                return .8 * y1(1);
            });

        rects.exit().remove();
    }
}