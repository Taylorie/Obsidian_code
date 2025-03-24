# 1. 后端

## 1. docker

### 1. docker 镜像全部启动命令 

```shell
docker start $(docker ps -a | awk '{ print $1}' | tail -n +2)
```

---

## 2. JSR303 校验

添加依赖

```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-validation</artifactId>
</dependency>
```

![image-20231206210212115](images/image-20231206210212115.png)

在实体类字段上添加对应的注解

```java
@Data
@ApiModel(value="AddCourseDto", description="新增课程基本信息")
public class AddCourseDto {

 @NotEmpty(message = "课程名称不能为空")
 @ApiModelProperty(value = "课程名称", required = true)
 private String name;

 @NotEmpty(message = "适用人群不能为空")
 @Size(message = "适用人群内容过少",min = 10)
 @ApiModelProperty(value = "适用人群", required = true)
 private String users;

 @ApiModelProperty(value = "课程标签")
 private String tags;
```

在方法形参上面添加注解开启校验

```java
@ApiOperation("新增课程基础信息")
@PostMapping("/course")
public CourseBaseInfoDto createCourseBase(@RequestBody @Validated AddCourseDto addCourseDto){
    //机构id，由于认证系统没有上线暂时硬编码
    Long companyId = 1L;
  return courseBaseInfoService.createCourseBase(companyId,addCourseDto);
}
```

如果需要捕获该校验异常需要添加

```java
@ResponseBody
@ExceptionHandler(MethodArgumentNotValidException.class)
@ResponseStatus(HttpStatus.INTERNAL_SERVER_ERROR)
public RestErrorResponse methodArgumentNotValidException(MethodArgumentNotValidException e) {
    BindingResult bindingResult = e.getBindingResult();
    List<String> msgList = new ArrayList<>();
    //将错误信息放在msgList
    bindingResult.getFieldErrors().stream().forEach(item->msgList.add(item.getDefaultMessage()));
    //拼接错误信息
    String msg = StringUtils.join(msgList, ",");
    log.error("【系统异常】{}",msg);
    return new RestErrorResponse(msg);
}
```

### 分组校验

有时候在同一个属性上设置一个校验规则不能满足要求，比如：订单编号由系统生成，在添加订单时要求订单编号为空，在更新 订单时要求订单编写不能为空。此时就用到了分组校验，同一个属性定义多个校验规则属于不同的分组，比如：添加订单定义@NULL规则属于insert分组，更新订单定义@NotEmpty规则属于update分组，insert和update是分组的名称，是可以修改的。

下边举例说明

我们用class类型来表示不同的分组，所以我们定义不同的接口类型（空接口）表示不同的分组，由于校验分组是公用的，所以定义在 base工程中。如下：

```java
public class ValidationGroups {
 public interface Insert{};
 public interface Update{};
 public interface Delete{};
}
```

实体类字段上指定分组信息

```java
@NotEmpty(groups = {ValidationGroups.Insert.class},message = "添加课程名称不能为空")
 @NotEmpty(groups = {ValidationGroups.Update.class},message = "修改课程名称不能为空")
// @NotEmpty(message = "课程名称不能为空")
 @ApiModelProperty(value = "课程名称", required = true)
 private String name;
```

在指定方法上指定使用的分组

```java
@ApiOperation("新增课程基础信息")
@PostMapping("/course")
public CourseBaseInfoDto createCourseBase(@RequestBody @Validated({ValidationGroups.Inster.class}) AddCourseDto addCourseDto){
    //机构id，由于认证系统没有上线暂时硬编码
    Long companyId = 1L;
  return courseBaseInfoService.createCourseBase(companyId,addCourseDto);
}

```



# 2. 浏览器

## 1. 跨域问题

同源策略是浏览器的一种安全机制，从一个地址请求另一个地址，如果协议、主机、端口三者全部一致则不属于跨域，否则有一个不一致就是跨域请求。

比如：

从http://localhost:8601 到  http://localhost:8602 由于端口不同，是跨域。

从http://192.168.101.10:8601 到  http://192.168.101.11:8601 由于主机不同，是跨域。

从http://192.168.101.10:8601 到  [https://192.168.101.10:8601](https://192.168.101.11:8601) 由于协议不同，是跨域。

注意：服务器之间不存在跨域请求。

浏览器判断是跨域请求会在请求头上添加origin，表示这个请求来源哪里。

比如：

```shell
Plaintext   GET / HTTP/1.1   Origin: http://localhost:8601  
```

服务器收到请求判断这个Origin是否允许跨域，如果允许则在响应头中说明允许该来源的跨域请求，如下：
```shell
Plaintext   Access-Control-Allow-Origin：http://localhost:8601  
```

如果允许任何域名来源的跨域请求，则响应如下：
```shell
Plaintext   Access-Control-Allow-Origin：*  
```

---

### 1. 后端解决跨域问题

新建配置类

```java
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.cors.CorsConfiguration;
import org.springframework.web.cors.UrlBasedCorsConfigurationSource;
import org.springframework.web.filter.CorsFilter;

/**
 * @author yu.zhang
 * @create 2023-11-25 23:11
 * @describe
 */
@Configuration
public class GlobalCorsConfig {

    /**
     * 允许跨域调用的过滤器
     */
    @Bean
    public CorsFilter corsFilter() {
        CorsConfiguration config = new CorsConfiguration();
        //允许白名单域名进行跨域调用
        config.addAllowedOrigin("*");
        //允许跨越发送cookie
        config.setAllowCredentials(true);
        //放行全部原始头信息
        config.addAllowedHeader("*");
        //允许所有请求方法跨域调用
        config.addAllowedMethod("*");
        UrlBasedCorsConfigurationSource source = new UrlBasedCorsConfigurationSource();
        source.registerCorsConfiguration("/**", config);
        return new CorsFilter(source);
    }
}
```

---

# 3. 数据库

## 1. MySQL 递归查询树形菜单

```sql
WITH [RECURSIVE]
     cte_name [(col_name [, col_name] ...)] AS (subquery)
     [, cte_name [(col_name [, col_name] ...)] AS (subquery)] ...
```

cte_name :公共表达式的名称,可以理解为表名,用来表示as后面跟着的子查询

col_name :公共表达式包含的列名,可以写也可以不写

```sql
with RECURSIVE t1  AS
(
  SELECT 1 as n
  UNION ALL
  SELECT n + 1 FROM t1 WHERE n < 5
)
SELECT * FROM t1;
```

```sql
with recursive t1 as (
select * from  course_category p where  id= '1'
union all
 select t.* from course_category t inner join t1 on t1.id = t.parentid
)
select *  from t1 order by t1.id, t1.orderby
```

