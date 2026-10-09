from django.utils.safestring import mark_safe
import copy

class Pagination(object):
    """分页组件"""
    """
    page: 当前页
    page_size: 每页显示多少条数据
    page_param: URL中页码的参数名，默认为page
    page_range: 显示当前页前后多少页，默认为5
    queryset: 需要分页的查询集
    total_page_count: 总页数
    start: 当前页的起始数据索引
    end: 当前页的结束数据索引
    page_queryset: 当前页的数据列表
    html: 生成分页HTML字符串的方法

    需要再html中使用分页组件时，调用Pagination类的html方法，生成分页HTML字符串，并将其传递给模板进行渲染:
    view.py:
    def user_list(request):
        queryset = models.UserInfo.objects.all()
        page_object = Pagination(request, queryset=queryset, page_size=2)
        context = {
            'queryset': page_object.page_queryset, # 获取当前页的数据
            'page_string': page_object.html() # 获取页码的HTML代码
        }

        return render(request, 'user_list.html', context)

    user_list.html:
    
    <ul class="pagination">
        {{ page_string }}
    </ul>

    """
    def __init__(self, request, queryset, page_size=10, page_param='page', page_range = 5):
        query_dict = copy.deepcopy(request.GET) # 获取当前URL中的查询字符串，并进行深复制，避免修改原来的查询字符串
        query_dict._mutable = True # 将查询字符串设置为可修改的
        self.query_dict = query_dict

        page = request.GET.get(page_param, '1')
        if page.isdecimal():
            page = int(page)
        else:
            page = 1
        
        self.page = page
        self.page_size = page_size

        self.start = (page - 1) * page_size
        self.end = page * page_size

        self.page_queryset = queryset[self.start:self.end]

        total_count = queryset.count()
        total_page_count, div = divmod(total_count, page_size)
        if div:
            total_page_count += 1

        self.total_page_count = total_page_count
        self.page_range = page_range
        self.page_param = page_param


    def html(self):
        start_page = max(int(self.page) - self.page_range, 1) # 计算起始页，不能小于1
        end_page = min(int(self.page) + self.page_range, self.total_page_count) # 计算结束页，不能大于总页数

        # 页码
        page_str_list = []

        # 首页
        self.query_dict.setlist(self.page_param, [1]) # 将查询字符串中的page参数设置为1
        first_page = '<li><a href="?{}">{}</a></li>'.format(self.query_dict.urlencode(), '首页')
        page_str_list.append(first_page)

        # 上一页
        self.query_dict.setlist(self.page_param, [max(1, int(self.page) - 1)])
        prev_page = '<li><a href="?{}">{}</a></li>'.format(self.query_dict.urlencode(), '上一页')
        page_str_list.append(prev_page)

        for i in range(start_page, end_page + 1):
            if i == int(self.page):
                temp = '<li class="active"><a href="?{}">{}</a></li>'.format(self.query_dict.urlencode(), i)
            else:
                self.query_dict.setlist(self.page_param, [i])
                temp = '<li><a href="?{}">{}</a></li>'.format(self.query_dict.urlencode(), i)
            page_str_list.append(temp)

        # 下一页
        self.query_dict.setlist(self.page_param, [min(self.total_page_count, int(self.page) + 1)])
        next_page = '<li><a href="?{}">{}</a></li>'.format(self.query_dict.urlencode(), '下一页')
        page_str_list.append(next_page)

        # 尾页
        self.query_dict.setlist(self.page_param, [self.total_page_count])
        last_page = '<li><a href="?{}">{}</a></li>'.format(self.query_dict.urlencode(), '尾页')
        page_str_list.append(last_page)

        # 搜索跳转
        search_page = """
        <li>    
            <form method="get" style="float: left; margin-left: 5px;">
                <input type="text" name="page" 
                    style="position: relative; float: left; display: inline-block; width: 80px; border-radius: 0;"
                    class="form-control" placeholder="页码">
                <button style="border-radius: 0;"  class="btn btn-default" type="submit">跳 转</button>
            </form>
        </li>

        """
        page_str_list.append(search_page)

        page_string = mark_safe("".join(page_str_list)) # 将字符串转换为HTML格式，避免被转义

        return page_string
