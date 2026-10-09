/**
 * 通用 AJAX 分页条渲染（依赖 jQuery + 全局 showAlert，均由 layout.html 提供）
 *
 * @param selector  分页条容器（jQuery 选择器，如 '#task-pagination'）
 * @param current   当前页码
 * @param totalPage 总页数；totalPage <= 1 时清空容器、不渲染分页条
 * @param loadFn    翻页回调，签名 function(page)，点页码/跳转时调用
 */
function renderPagination(selector, current, totalPage, loadFn) {
    var $box = $(selector);
    if (totalPage < 1) { $box.empty(); return; }

    var startPage = Math.max(current - 5, 1);
    var endPage = Math.min(current + 5, totalPage);

    var html = '<ul class="pagination pagination-sm justify-content-center">';
    // 首页 / 上一页
    html += '<li class="page-item' + (current === 1 ? ' disabled' : '') + '"><a class="page-link" href="javascript:;" data-page="1">首页</a></li>';
    html += '<li class="page-item' + (current === 1 ? ' disabled' : '') + '"><a class="page-link" href="javascript:;" data-page="' + (current - 1) + '">上一页</a></li>';
    // 页码窗口（当前页高亮）
    for (var i = startPage; i <= endPage; i++) {
        html += '<li class="page-item' + (i === current ? ' active' : '') + '"><a class="page-link" href="javascript:;" data-page="' + i + '">' + i + '</a></li>';
    }
    // 下一页 / 尾页
    html += '<li class="page-item' + (current === totalPage ? ' disabled' : '') + '"><a class="page-link" href="javascript:;" data-page="' + (current + 1) + '">下一页</a></li>';
    html += '<li class="page-item' + (current === totalPage ? ' disabled' : '') + '"><a class="page-link" href="javascript:;" data-page="' + totalPage + '">尾页</a></li>';
    // 跳转框
    html += '<li class="page-item"><span class="d-flex align-items-center ms-2" style="gap:4px;">' +
        '<input type="text" class="form-control form-control-sm" placeholder="页码" style="width:70px;">' +
        '<button class="btn btn-outline-secondary btn-sm jump-btn" type="button">跳转</button>' +
        '</span></li>';
    html += '</ul>';
    $box.html(html);

    // 页码点击（事件委托；disabled 不响应）
    $box.off('click', 'a.page-link').on('click', 'a.page-link', function (e) {
        e.preventDefault();
        if ($(this).closest('li.page-item').hasClass('disabled')) return;
        var p = parseInt($(this).data('page'), 10);
        if (!isNaN(p)) loadFn(p);
    });
    // 跳转框：点按钮
    $box.off('click', '.jump-btn').on('click', '.jump-btn', function () {
        var v = parseInt($box.find('input').val(), 10);
        if (isNaN(v) || v < 1 || v > totalPage) { showAlert('请输入 1 ~ ' + totalPage + ' 之间的页码'); return; }
        loadFn(v);
    });
    // 跳转框：回车
    $box.off('keydown', 'input').on('keydown', 'input', function (e) {
        if (e.keyCode === 13) { e.preventDefault(); $box.find('.jump-btn').trigger('click'); }
    });
}
