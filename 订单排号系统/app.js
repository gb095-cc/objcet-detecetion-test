// app.js
App({
  onLaunch(){
    wx.cloud.init({
      //云开发环境id不是小程序appid，位置在云开发控制台找到，也可以在开通云开发环境时，微信团队发的消息中找到
      env:'cloud-learn-5ggidkk310b35ee3'
    })
  }
})
