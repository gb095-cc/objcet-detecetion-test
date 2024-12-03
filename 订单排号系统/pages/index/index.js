let paihaoDB = wx.cloud.database().collection('paihao');
Page({
  data: {
    userId: '', // 用于存储用户唯一标识符
    paihaoxinxi: ''
  },

  onLoad() {
    this.getUserId(); // 获取用户 ID
    this.getNum();
  },

  // 获取用户 ID（假设用 openid 作为唯一标识）
  getUserId() {
    wx.cloud.callFunction({
      name: 'getOpenId', // 云函数获取 openid
      success: res => {
        this.setData({
          userId: res.result.openid // 保存用户 ID
        });
      },
      fail: err => {
        console.error('获取用户 ID 失败', err);
      }
    });
  },

  // 排号
  paihao() {
    console.log("用户点击了排号");
    let key = this.getNianYueRi();
    let paihao = null;
    console.log(key);
    
    // 查询当前用户已经排号到多少位
    paihaoDB.where({
      key: key,
      userId: this.data.userId // 根据用户 ID 查询
    })
    .count()
    .then(res => {
      console.log('查询成功', res);
      paihao = res.total + 1;

      // 添加排号信息到数据库
      paihaoDB.add({
        data: {
          key: key,
          paihao: paihao,
          userId: this.data.userId // 保存用户 ID
        }
      }).then(res => {
        console.log('添加成功', res);
        this.getNum();
        // 跳转到订单状态页面
        wx.navigateTo({
          url: `/pages/orderStatus/orderStatus?paihao=${paihao}` // 传递用户号码
        });
      }).catch(res => {
        console.log('添加失败', res);
      });
    })
    .catch(res => {
      console.log('查询失败', res);
    });
  },

  // 查看自己的排号信息
  getNum() {
    paihaoDB.where({
      userId: this.data.userId // 根据用户 ID 查询
    }).get()
    .then(res => {
      let len = res.data.length;
      if (len > 0) {
        let obj = res.data[len - 1];
        console.log('获取排号信息成功', obj.paihao);
        this.setData({
          paihaoxinxi: '您当前的号码是' + obj.paihao
        });
      } else {
        this.setData({
          paihaoxinxi: '您还未排号'
        });
      }
    })
    .catch(res => {
      console.log('获取排号信息失败', res);
    });
  },

  // 获取当前的年月日
  getNianYueRi() {
    let date = new Date();
    let year = date.getFullYear();
    let month = date.getMonth() + 1;
    let day = date.getDate();
    let key = '' + year + month + day;
    return key;
  }
});

