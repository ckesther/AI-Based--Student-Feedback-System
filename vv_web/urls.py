"""vv_web URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import path,include
from .views import result
# add this to import our views file

from vv_web import views
urlpatterns = [
    path('admin/', admin.site.urls),    # add these to configure our home page (default view) and result web page
    path('',views.home,name='home'),
    path('result/',views.result,name='result'),
    path('resultp/', views.resultp, name='resultp'),
    path('resultj/', views.resultj, name='resultj'),
    path('dd/',views.dd,name='dd'),
    path('ddp/', views.ddp, name='ddp'),
    path('ddj/', views.ddj, name='ddj'),
    path('any/', views.any, name='any'),
    path('anyc/',views.anyc,name='anyc'),
    path('anyp/', views.anyp, name='anyp'),
    path('anyj/', views.anyj, name='anyj'),
    path('hom/',views.hom,name='hom'),
    path('inx/',views.inx,name='inx'),
    path('inx1/', views.inx1, name='inx1'),
    path('inx2/', views.inx2, name='inx2'),
    path('inx3/', views.inx3, name='inx3'),
    path('pie/',views.pie,name='pie'),
    path('piep/', views.piep, name='piep'),
    path('piej/', views.piej, name='piej'),
    path('login/',views.login,name='login'),
    path('contact/',views.contact,name='contact'),
    path('thanks/',views.thanks,name='thanks'),
    path('thnx/',views.thnx,name='thnx'),
    path('txtrpt',views.txtrpt,name='txtrpt'),
    path('txtrpt_p', views.txtrpt_p, name='txtrpt_p'),
    path('txtrpt_j', views.txtrpt_j, name='txtrpt_j'),
    path('loginstd/',views.loginstd,name='loginstd')
]