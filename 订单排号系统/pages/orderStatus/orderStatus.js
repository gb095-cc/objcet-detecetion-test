const orderDB = wx.cloud.database().collection('orders'); // 假设你的数据库集合是 orders

Page({
  data: {
    userPaihao: '', // 用户号码
    currentPreparing: 0, // 当前正在准备的号码
    waitingTime: 0 // 等待时间
  },

  onLoad(options) {
    // 获取用户号码并初始化状态
    this.setData({
      userPaihao: options.paihao // 获取用户号码
    });
    this.refreshStatus(); // 加载初始状态
  },

  // 刷新状态
  refreshStatus() {
    orderDB.orderBy('createTime', 'asc').get().then(res => {
      if (res.data.length > 0) {
        const preparingOrder = res.data[0]; // 假设准备的第一个订单
        this.setData({
          currentPreparing: preparingOrder.paihao, // 当前准备的号码
          waitingTime: (preparingOrder.paihao - this.data.userPaihao) * 5 // 假设每人预计5分钟
        });
      } else {
        // 如果没有订单，显示默认信息
        this.setData({
          currentPreparing: '暂无',
          waitingTime: '0'
        });
      }
    }).catch(err => {
      console.error('获取状态失败', err);
    });
  },

  // 返回按钮处理
  goBack() {
    wx.navigateBack(); // 返回到上一个页面
  }
});
